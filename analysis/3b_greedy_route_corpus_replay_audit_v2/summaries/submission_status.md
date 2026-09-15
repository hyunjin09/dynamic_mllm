STOPPED by user on2026-09-15 11:41:56 KST: replay2994 and dependent report2957
cancelled. Saved outputs are preserved; no restart is authorized. V2 remains
incomplete. Evidence: concurrency_v1/user_cancellation.json. Details below
describe the state before cancellation.

Dense2954 and certification2955 COMPLETED successfully. All10,000 dense
records certified:5,702 current Correct,4,298 current Wrong. Old→current:
5,607 C→C,125 C→W,95 W→C,4,173 W→W;220 labels changed(2.2%).

Routed2994 is running on8GPUs with3 processes/GPU since2026-09-15 11:23:14 KST.
It replaces2956 under the user's explicit concurrency request, preserving
720,540 durable records and1,834 complete samples. Final report2957 now waits
afterok:2994; its dependency was retargeted before2956 was stopped.

Original inference contracts and completed records remain unchanged. Versioned
scheduling evidence is under concurrency_v1/. Concurrent32-anchor parity,
24-worker sparse repeatability and all9,323 routes in24 completed pilot samples
pass exactly. Pilot throughput34.52routes/s; expensive-first DocVQA production
17.53routes/s over433.8seconds. At11:36:42 KST,8,414new routes and8complete
production samples passed monitoring and independent record/source/dense checks.
No errors;96–99%GPU use,25–26GB/GPU. Preliminary remaining replay ETA25–55hours
(Sep16 afternoon–Sep17 evening KST); report runtime and final review additional.
Details: concurrency_v1/initial_throughput_review.md. Brief assistant monitoring
is now paused;2994 continues and2957 remains dependent. Final RS/readiness
review remains pending.
