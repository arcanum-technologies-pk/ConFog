"""Four-validator permissioned Hyperledger Besu network (QBFT consensus) in Docker, for the ledger evaluation.

    python3 scripts/besu_net.py up      # generate keys + genesis, start 4 validators, wait for blocks
    python3 scripts/besu_net.py down    # stop and remove the containers and the network

Permissioning: only the 4 validator nodes may join (node allowlist) and only the 10 member accounts
may send transactions (account allowlist); one extra funded "outsider" account is NOT allowlisted and is
used by the security test. The keys are deterministic test keys of this local network only.
RPC of validator 1 on http://127.0.0.1:8546 (Hardhat keeps 8545).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

IMAGE = os.environ.get("BESU_IMAGE", "hyperledger/besu:24.9.1")
DIR = Path(__file__).resolve().parents[1] / "blockchain" / "besu"
NET, SUBNET, IPS = "confog-besu", "172.30.0.0/24", [f"172.30.0.{11 + i}" for i in range(4)]
RPC = "http://127.0.0.1:8546"
BLOCK_S = 1


def sh(*a, check=True, **kw):
    return subprocess.run(list(a), check=check, text=True, capture_output=True, **kw)


def rpc(method, params=None):
    req = urllib.request.Request(RPC, json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                                                   "params": params or []}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3) as r:
        return json.loads(r.read())["result"]


def keys():
    from eth_account import Account
    from eth_utils import keccak
    members = [Account.from_key(keccak(text=f"confog-member-{i}")) for i in range(10)]
    outsider = Account.from_key(keccak(text="confog-outsider"))
    return members, outsider


def down():
    for i in range(4):
        sh("docker", "rm", "-f", f"besu-{i + 1}", check=False)
    sh("docker", "network", "rm", NET, check=False)
    print("besu: stopped")


def up():
    down()
    members, outsider = keys()
    if DIR.exists():
        shutil.rmtree(DIR)
    DIR.mkdir(parents=True)
    (DIR / "accounts.json").write_text(json.dumps({"members": [a.key.hex() for a in members],
                                                   "outsider": outsider.key.hex()}, indent=1))
    alloc = {a.address[2:].lower(): {"balance": "0xad78ebc5ac6200000"} for a in members + [outsider]}
    cfg = {"genesis": {"config": {"chainId": 1337, "berlinBlock": 0, "londonBlock": 0, "zeroBaseFee": True,
                                  "qbft": {"blockperiodseconds": BLOCK_S, "epochlength": 30000,
                                           "requesttimeoutseconds": 4}},
                       "nonce": "0x0", "timestamp": "0x58ee40ba", "gasLimit": "0x1fffffffffffff",
                       "difficulty": "0x1", "coinbase": "0x0000000000000000000000000000000000000000",
                       "mixHash": "0x63746963616c2062797a616e74696e65206661756c7420746f6c6572616e6365",
                       "alloc": alloc},
           "blockchain": {"nodes": {"generate": True, "count": 4}}}
    (DIR / "qbftConfigFile.json").write_text(json.dumps(cfg, indent=1))
    uid = f"{os.getuid()}:{os.getgid()}"
    print(f"besu: pulling {IMAGE} and generating validator keys")
    sh("docker", "pull", IMAGE)
    sh("docker", "run", "--rm", "--user", uid, "-v", f"{DIR}:/data", IMAGE, "operator",
       "generate-blockchain-config", "--config-file=/data/qbftConfigFile.json", "--to=/data/networkFiles",
       "--private-key-file-name=key")
    vdirs = sorted((DIR / "networkFiles" / "keys").iterdir())
    assert len(vdirs) == 4, vdirs
    enodes = [f"enode://{(d / 'key.pub').read_text().strip().removeprefix('0x')}@{ip}:30303"
              for d, ip in zip(vdirs, IPS)]
    perm = DIR / "perm"
    perm.mkdir()
    (perm / "perm.toml").write_text(
        "accounts-allowlist=[" + ",".join(f'"{a.address.lower()}"' for a in members) + "]\n"
        "nodes-allowlist=[" + ",".join(f'"{e}"' for e in enodes) + "]\n")
    shutil.copy(DIR / "networkFiles" / "genesis.json", DIR / "genesis.json")
    sh("docker", "network", "create", "--subnet", SUBNET, NET)
    for i, (d, ip) in enumerate(zip(vdirs, IPS)):
        args = ["docker", "run", "-d", "--name", f"besu-{i + 1}", "--user", uid, "--network", NET, "--ip", ip,
                "-v", f"{d}:/opt/node:ro", "-v", f"{DIR / 'genesis.json'}:/opt/genesis.json:ro",
                "-v", f"{perm}:/opt/perm"]
        if i == 0:
            args += ["-p", "127.0.0.1:8546:8545"]
        args += [IMAGE, "--data-path=/tmp/besu", "--genesis-file=/opt/genesis.json",
                 "--node-private-key-file=/opt/node/key", f"--p2p-host={ip}", "--p2p-port=30303",
                 "--rpc-http-enabled", "--rpc-http-host=0.0.0.0", "--rpc-http-port=8545",
                 "--rpc-http-api=ETH,NET,WEB3,QBFT,TXPOOL,PERM", "--host-allowlist=*",
                 "--rpc-http-cors-origins=all", "--min-gas-price=0",
                 "--permissions-nodes-config-file-enabled", "--permissions-nodes-config-file=/opt/perm/perm.toml",
                 "--permissions-accounts-config-file-enabled",
                 "--permissions-accounts-config-file=/opt/perm/perm.toml"]
        if i > 0:
            args += [f"--bootnodes={enodes[0]}"]
        sh(*args)
    print("besu: 4 validators started, waiting for blocks ...")
    for _ in range(180):
        try:
            n = int(rpc("eth_blockNumber"), 16)
            if n >= 3:
                v = rpc("qbft_getValidatorsByBlockNumber", ["latest"])
                peers = int(rpc("net_peerCount"), 16)
                print(f"besu: block {n}, {len(v)} validators, {peers} peers of validator 1 -> ready on {RPC}")
                (DIR / "network.json").write_text(json.dumps(
                    {"image": IMAGE, "consensus": "QBFT", "validators": len(v), "block_period_s": BLOCK_S,
                     "node_allowlist": len(enodes), "account_allowlist": len(members)}, indent=1))
                return
        except Exception:
            pass
        time.sleep(1)
    print(sh("docker", "logs", "--tail", "40", "besu-1", check=False).stdout)
    sys.exit("besu: no blocks after 180 s - see the log above")


if __name__ == "__main__":
    {"up": up, "down": down}[(sys.argv[1:] or ["up"])[0]]()
