// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import "./AccessPolicy.sol";
import "./DeviceRegistry.sol";

/// @title AlertAnchor - stores only hashes of alerts (alerts stay off-chain, no patient data on-chain).
/// An alert is accepted only from an active device whose reported firmware/model hash is attested.
contract AlertAnchor {
    struct Anchor {
        bytes32 deviceId;
        uint64 timestamp;
        uint32 count;      // 1 for a single alert, n for a Merkle batch
    }

    AccessPolicy public immutable policy;
    DeviceRegistry public immutable registry;
    mapping(bytes32 => Anchor) public anchors;   // alert hash or Merkle root -> anchor

    event Anchored(bytes32 indexed hash, bytes32 indexed deviceId, uint32 count);

    constructor(AccessPolicy _policy, DeviceRegistry _registry) {
        policy = _policy;
        registry = _registry;
    }

    function _anchor(bytes32 h, bytes32 deviceId, bytes32 fwHash, uint32 count) internal {
        require(policy.isWriter(msg.sender), "not authorised");
        require(registry.isActive(deviceId), "device not active");
        require(registry.attest(deviceId, fwHash), "firmware not attested");
        require(anchors[h].timestamp == 0, "already anchored");
        anchors[h] = Anchor(deviceId, uint64(block.timestamp), count);
        emit Anchored(h, deviceId, count);
    }

    function anchor(bytes32 alertHash, bytes32 deviceId, bytes32 fwHash) external {
        _anchor(alertHash, deviceId, fwHash, 1);
    }

    function anchorBatch(bytes32 merkleRoot, bytes32 deviceId, bytes32 fwHash, uint32 count) external {
        _anchor(merkleRoot, deviceId, fwHash, count);
    }

    function isAnchored(bytes32 h) external view returns (bool) {
        return anchors[h].timestamp != 0;
    }
}
