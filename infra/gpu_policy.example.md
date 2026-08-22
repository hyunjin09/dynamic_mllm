# Machine-Local Compute Policy Template

Copy this file to the Git-ignored `infra/gpu_policy.md` and document the
current server's actual execution rules.

Define at least:

- whether CPU work runs locally or through a scheduler;
- how GPU allocations are acquired;
- physical GPU count and type;
- CPU, memory, time, and concurrency limits;
- persistent and scratch storage locations;
- container or module requirements;
- commands for inspecting live capacity and jobs.

Do not copy node names, partitions, storage roots, or resource assumptions
from another server without verifying them live.
