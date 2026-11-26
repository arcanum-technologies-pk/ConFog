// Local Ethereum-compatible network for the experiments (permissioning is enforced by AccessPolicy.sol).
module.exports = {
  networks: {
    hardhat: { chainId: 31337 },   // automine: one block per transaction
  },
};
