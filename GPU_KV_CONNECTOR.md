# GPU-KV Connector

The GPU-KV connector is maintained as a first-class application in the
[GPU-KV repository](https://github.com/ScaleX-IO/GPU-KV/tree/feature/late-bound-superrequests/applications/vllm).

This vLLM branch retains only the small core compatibility change that lets an
external connector request an aligned cross-layer KV-cache allocation. Install
the connector from the GPU-KV checkout by following its
`applications/vllm/README.md`.
