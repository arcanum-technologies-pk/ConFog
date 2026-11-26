"""Phase IV - permissioned blockchain for device trust and alert provenance.

Deploys AccessPolicy, DeviceRegistry and AlertAnchor on a local Ethereum-compatible node (Hardhat),
then measures gas / latency per operation and runs security tests:
  tamper detection of off-chain alerts, unregistered / revoked device, unauthorised writer,
  firmware attestation mismatch, replay (re-anchoring the same alert).
Only SHA-256 hashes (or Merkle roots) of alerts go on-chain; alert content stays off-chain.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import time

import numpy as np

from .common import ROOT, log, save_json

CHAIN_DIR = ROOT / "blockchain"


# ============================================================================ hashing / Merkle
def h256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def alert_hash(alert: dict) -> bytes:
    """Keyed fingerprint (HMAC-SHA-256) with a secret of the hospital, so stored values cannot be
    brute-forced from known record formats. PATCH_HMAC"""
    import hmac, os
    key = os.environ.get("HOSPITAL_KEY", "confog-demo-key").encode()
    return hmac.new(key, json.dumps(alert, sort_keys=True, separators=(",", ":")).encode(), "sha256").digest()


def merkle_root(leaves: list[bytes]) -> bytes:
    level = list(leaves)
    while len(level) > 1:
        nxt = [h256(min(a, b) + max(a, b)) for a, b in zip(level[0::2], level[1::2])]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def merkle_proof(leaves: list[bytes], i: int) -> list[bytes]:
    proof, level = [], list(leaves)
    while len(level) > 1:
        j = i ^ 1
        if j < len(level):
            proof.append(level[j])
        nxt = [h256(min(a, b) + max(a, b)) for a, b in zip(level[0::2], level[1::2])]
        if len(level) % 2:
            nxt.append(level[-1])
        level, i = nxt, i // 2
    return proof


def merkle_verify(leaf: bytes, proof: list[bytes], root: bytes) -> bool:
    h = leaf
    for p in proof:
        h = h256(min(h, p) + max(h, p))
    return h == root


def fog_firmware_hash(bundle_dir) -> tuple[bytes, list[str]]:
    """SHA-256 over the fog node's actual software: model files + runtime code (what the node runs)."""
    files = [bundle_dir / "fog_lgbm.txt", bundle_dir / "fog_bundle.pkl",
             ROOT / "src" / "fog_runtime.py", ROOT / "src" / "fog_node.py"]
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\0" + f.read_bytes())
    return h.digest(), [str(f) for f in files]


# ============================================================================ node management
def _port_open(url: str) -> bool:
    host, port = url.split("//")[1].split(":")
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, int(port))) == 0


def _ensure_node(rpc: str):
    if not (CHAIN_DIR / "node_modules").exists():
        log.info("chain: installing pinned hardhat + solc (npm ci)")
        subprocess.run(["npm", "install", "--no-audit", "--no-fund"], cwd=CHAIN_DIR, check=True)
    subprocess.run(["node", "compile.js"], cwd=CHAIN_DIR, check=True, capture_output=True)   # always current ABI
    if _port_open(rpc):
        return None
    log.info("chain: starting local Hardhat node")
    env = dict(os.environ, HARDHAT_DISABLE_TELEMETRY_PROMPT="true")
    proc = subprocess.Popen([shutil.which("npx") or "npx", "hardhat", "node", "--hostname", "127.0.0.1",
                             "--port", rpc.rsplit(":", 1)[1]], cwd=CHAIN_DIR, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(120):
        if _port_open(rpc):
            return proc
        time.sleep(0.5)
    proc.kill()
    raise RuntimeError("Hardhat node did not start")


# ============================================================================ experiment
def _besu_keys():   # PATCH_BESU
    k = json.loads((CHAIN_DIR / "besu" / "accounts.json").read_text())
    return k["members"], k["outsider"]


class Chain:
    def __init__(self, rpc: str, keys=None):
        from web3 import Web3
        self.w3 = Web3(Web3.HTTPProvider(rpc))
        self.outsider = None
        if keys is None:                       # Hardhat: unlocked accounts
            self.acc = self.w3.eth.accounts
        else:                                  # Besu: transactions signed locally with the member keys
            from eth_account import Account
            members, outsider = keys
            accts = [Account.from_key(k) for k in members]
            out = Account.from_key(outsider)
            try:
                from web3.middleware import ExtraDataToPOAMiddleware as _poa
            except ImportError:
                from web3.middleware import geth_poa_middleware as _poa
            self.w3.middleware_onion.inject(_poa, layer=0)
            try:
                from web3.middleware import SignAndSendRawMiddlewareBuilder
                self.w3.middleware_onion.inject(SignAndSendRawMiddlewareBuilder.build(accts + [out]), layer=0)
            except ImportError:
                from web3.middleware import construct_sign_and_send_raw_middleware
                self.w3.middleware_onion.add(construct_sign_and_send_raw_middleware(accts + [out]))
            self.acc = [a.address for a in accts]
            self.outsider = out.address
        self.admin = self.acc[0]
        self.stats: dict[str, list] = {}

    def deploy(self, name: str, *args):
        art = json.loads((CHAIN_DIR / "build" / f"{name}.json").read_text())
        c = self.w3.eth.contract(abi=art["abi"], bytecode=art["bytecode"])
        rc = self._send(f"deploy_{name}", c.constructor(*args), self.admin)
        return self.w3.eth.contract(address=rc.contractAddress, abi=art["abi"])

    def _send(self, op: str, fn, sender):
        t0 = time.perf_counter()
        tx = fn.transact({"from": sender})
        rc = self.w3.eth.wait_for_transaction_receipt(tx)
        self.stats.setdefault(op, []).append((rc.gasUsed, 1000 * (time.perf_counter() - t0)))
        return rc

    def tx(self, op, fn, sender=None):
        return self._send(op, fn, sender or self.admin)

    def transfer_rejected(self, sender) -> bool:
        """True if a plain value transfer from `sender` is refused by the network (account permissioning)."""
        self.last_error = None    # PATCH_BESU2: zero value, so only permissioning can refuse it
        try:
            tx = self.w3.eth.send_transaction({"from": sender, "to": self.admin, "value": 0})
            return self.w3.eth.wait_for_transaction_receipt(tx, timeout=30).status != 1
        except Exception as e:
            self.last_error = str(e)
            return True

    def reverts(self, fn, sender=None) -> bool:
        try:
            fn.transact({"from": sender or self.admin})
            return False
        except Exception:
            return True


def run_chain(cfg: dict) -> None:
    out = cfg["paths"]["results"] / "chain" / "chain_results.json"
    if out.exists():
        log.info("chain: results exist - skip")
        return
    c = cfg["blockchain"]
    alerts_f = cfg["paths"]["results"] / "explain" / "alerts.jsonl"
    alerts = [json.loads(l) for l in alerts_f.read_text().splitlines()] if alerts_f.exists() else []
    rng = np.random.default_rng(0)
    while len(alerts) < c["n_alerts"]:   # top up with clearly marked synthetic alerts if few were forwarded
        alerts.append({"alert_id": f"S{len(alerts):07d}", "synthetic": True, "device_id": f"FOG-{len(alerts) % 3 + 1:02d}",
                       "fog_score": float(rng.random())})
    alerts = alerts[: c["n_alerts"]]

    besu = c.get("node") == "besu"
    if besu:   # external 4-validator network started by scripts/besu_net.py
        subprocess.run(["node", "compile.js"], cwd=CHAIN_DIR, check=True, capture_output=True)
        proc = None
    else:
        proc = _ensure_node(c["rpc"])
    try:
        ch = Chain(c["rpc"], keys=_besu_keys() if besu else None)
        policy = ch.deploy("AccessPolicy")
        reg = ch.deploy("DeviceRegistry", policy.address)
        anc = ch.deploy("AlertAnchor", policy.address, reg.address)
        fog_nodes = ch.acc[1:4]
        for a in fog_nodes:
            ch.tx("set_writer", policy.functions.setWriter(a, True))

        # devices: 3 fog gateways (firmware hash = hash of the real fog software and model)
        # + IoMT device identities, each registered with its firmware image hash
        mdir = cfg["paths"]["models"] / "fog_main"
        fog_fw, fog_files = fog_firmware_hash(mdir)
        dev_ids = [f"FOG-{i:02d}" for i in (1, 2, 3)] + [f"IOMT-{i:04d}" for i in range(c["n_devices"] - 3)]
        did = {d: h256(d.encode()) for d in dev_ids}
        fw = {d: (fog_fw if d.startswith("FOG") else h256(f"iomt-firmware-v1:{d}".encode())) for d in dev_ids}
        for d in dev_ids:
            ch.tx("register_device", reg.functions.register(did[d], fw[d]), fog_nodes[0])
        log.info(f"chain: {len(dev_ids)} devices registered")

        # (a) per-alert anchoring
        leaves = [alert_hash(a) for a in alerts]
        half = len(alerts) // 2
        for a, h in zip(alerts[:half], leaves[:half]):
            ch.tx("anchor_alert", anc.functions.anchor(h, did[a["device_id"]], fw[a["device_id"]]), fog_nodes[0])
        # (b) Merkle-batch anchoring of the other half
        batches = []
        for i in range(half, len(alerts), c["batch_size"]):
            lv = leaves[i:i + c["batch_size"]]
            root = merkle_root(lv)
            ch.tx("anchor_batch", anc.functions.anchorBatch(root, did["FOG-01"], fw["FOG-01"], len(lv)), fog_nodes[0])
            batches.append((i, lv, root))
        log.info(f"chain: {half} alerts anchored individually, {len(batches)} Merkle batches")

        # ---- verification of untouched alerts (must all pass) ----
        def verify(k: int, alert: dict) -> bool:
            h = alert_hash(alert)
            if k < half:
                return anc.functions.isAnchored(h).call()
            for i, lv, root in batches:
                if i <= k < i + len(lv):
                    return anc.functions.isAnchored(root).call() and merkle_verify(h, merkle_proof(lv, k - i), root)
            return False
        ok = sum(verify(k, a) for k, a in enumerate(alerts))

        # ---- tamper test: modify stored alerts off-chain, they must fail verification ----
        idx = rng.choice(len(alerts), min(c["n_tamper"], len(alerts)), replace=False)
        detected = 0
        for k in idx:
            t = dict(alerts[k])
            t["fog_score"] = round(float(t.get("fog_score", 0)) * 0.5, 6)   # e.g. an insider lowering a score
            detected += not verify(int(k), t)

        # ---- negative security tests (each must revert / return false) ----
        ghost = h256(b"UNREGISTERED")
        ch.tx("revoke_device", reg.functions.revoke(did["IOMT-0000"]), fog_nodes[0])
        # a fog node whose model file was modified (e.g. an insider swapping the detector) reports a new hash
        tampered = hashlib.sha256(fog_fw + b"model-file-modified").digest()
        a0 = alerts[0]["device_id"]
        tests = {
            "unregistered_device_rejected": ch.reverts(anc.functions.anchor(h256(b"x1"), ghost, fog_fw), fog_nodes[0]),
            "revoked_device_rejected": ch.reverts(
                anc.functions.anchor(h256(b"x2"), did["IOMT-0000"], fw["IOMT-0000"]), fog_nodes[0]),
            "unauthorised_writer_rejected": ch.reverts(
                anc.functions.anchor(h256(b"x3"), did["FOG-01"], fog_fw), ch.acc[9]),
            "replay_rejected": ch.reverts(anc.functions.anchor(leaves[0], did[a0], fw[a0]), fog_nodes[0]),
            "tampered_fog_model_rejected": ch.reverts(
                anc.functions.anchor(h256(b"x4"), did["FOG-02"], tampered), fog_nodes[0]),
            "firmware_mismatch_detected": not reg.functions.attest(did["IOMT-0001"], h256(b"malicious-fw")).call(),
            "firmware_match_accepted": reg.functions.attest(did["IOMT-0001"], fw["IOMT-0001"]).call(),
        }
        if besu:   # network-level permissioning: non-member account refused, member accepted
            rej = ch.transfer_rejected(ch.outsider)
            reason = ch.last_error or ""
            tests["non_member_account_rejected"] = rej and ("not authorized" in reason.lower() or "permission" in reason.lower())
            log.info(f"chain: non-member refusal reason: {reason[:160]}")
            tests["member_account_accepted"] = not ch.transfer_rejected(ch.acc[5])
        ops = {op: {"n": len(v), "gas_mean": float(np.mean([g for g, _ in v])),
                    "latency_ms_mean": float(np.mean([t for _, t in v])),
                    "latency_ms_p95": float(np.percentile([t for _, t in v], 95))} for op, v in ch.stats.items()}
        n_batched = len(alerts) - half
        per_alert_gas_batch = (sum(g for g, _ in ch.stats["anchor_batch"]) / n_batched) if n_batched else None
        res = {"operations": ops, "n_alerts": len(alerts), "n_real_alerts": sum("synthetic" not in a for a in alerts),
               "verified_untampered": int(ok), "tampered": int(len(idx)), "tamper_detected": int(detected),
               "tamper_detection_rate": detected / max(len(idx), 1), "security_tests": tests,
               "gas_per_alert_single": ops["anchor_alert"]["gas_mean"], "gas_per_alert_batched": per_alert_gas_batch,
               "on_chain_bytes_per_alert": 32, "node": "Hardhat 2.22.17 (local, automine)",
               "fog_firmware_sha256": fog_fw.hex(), "fog_firmware_files": fog_files}
        if besu:
            net = json.loads((CHAIN_DIR / "besu" / "network.json").read_text())
            res["node"] = (f"Hyperledger Besu ({net['image']}), {net['consensus']}, {net['validators']} validators, "
                           f"{net['block_period_s']} s blocks, node and account permissioning")
        save_json(res, out)
        log.info(f"chain: verified {ok}/{len(alerts)} untampered | tamper detected {detected}/{len(idx)} | "
                 f"gas/alert single {res['gas_per_alert_single']:.0f} vs batched {per_alert_gas_batch:.0f} | "
                 f"tests {tests}")
    finally:
        if proc is not None:
            os.killpg(proc.pid, signal.SIGTERM)   # npx + node child
