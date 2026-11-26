// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

/// @title AccessPolicy - permissioned write access for hospital nodes (admin grants / revokes writers)
contract AccessPolicy {
    address public immutable admin;
    mapping(address => bool) public writers;

    event WriterSet(address indexed node, bool allowed);

    constructor() {
        admin = msg.sender;
        writers[msg.sender] = true;
    }

    modifier onlyAdmin() {
        require(msg.sender == admin, "not admin");
        _;
    }

    function setWriter(address node, bool allowed) external onlyAdmin {
        writers[node] = allowed;
        emit WriterSet(node, allowed);
    }

    function isWriter(address node) external view returns (bool) {
        return writers[node];
    }
}
