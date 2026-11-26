// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import "./AccessPolicy.sol";

/// @title DeviceRegistry - onboarding, revocation and firmware attestation of fog nodes / IoMT devices
contract DeviceRegistry {
    enum Status { None, Active, Revoked }

    struct Device {
        bytes32 firmwareHash;
        Status status;
        uint64 registeredAt;
    }

    AccessPolicy public immutable policy;
    mapping(bytes32 => Device) public devices;

    event Registered(bytes32 indexed id, bytes32 firmwareHash);
    event Revoked(bytes32 indexed id);
    event FirmwareUpdated(bytes32 indexed id, bytes32 firmwareHash);

    constructor(AccessPolicy _policy) {
        policy = _policy;
    }

    modifier onlyWriter() {
        require(policy.isWriter(msg.sender), "not authorised");
        _;
    }

    function register(bytes32 id, bytes32 firmwareHash) external onlyWriter {
        require(devices[id].status == Status.None, "already registered");
        devices[id] = Device(firmwareHash, Status.Active, uint64(block.timestamp));
        emit Registered(id, firmwareHash);
    }

    function revoke(bytes32 id) external onlyWriter {
        require(devices[id].status == Status.Active, "not active");
        devices[id].status = Status.Revoked;
        emit Revoked(id);
    }

    function updateFirmware(bytes32 id, bytes32 firmwareHash) external onlyWriter {
        require(devices[id].status == Status.Active, "not active");
        devices[id].firmwareHash = firmwareHash;
        emit FirmwareUpdated(id, firmwareHash);
    }

    function isActive(bytes32 id) external view returns (bool) {
        return devices[id].status == Status.Active;
    }

    /// @notice true if the device is active and reports the registered firmware hash
    function attest(bytes32 id, bytes32 firmwareHash) external view returns (bool) {
        Device storage d = devices[id];
        return d.status == Status.Active && d.firmwareHash == firmwareHash;
    }
}
