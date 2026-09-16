# Adaptive rank128 stopped to correct experiment order

The user clarified: “we were supposed to be training the rank64 .. the scheduled follow up was wrong”. The adaptive rank128 continuation had been incorrectly placed before rank64. The automation now selects G32rank64fixed immediately and does not require F completion or text assessment.

F was stopped on macm3 at2026-09-15T22:04:28UTC:2,630 observed updates,2,600 durable updates,2completeepochs. This is an explicit ordering correction, not a plateau decision or completion of the44-epoch recipe. Dispatcher13662 and trainer13675 were verified by birth/command/parentage before SIGTERM; both exited. Original source bindings remain intact.

Both retained selectors point to update2,600: validation CE3.7786049545056493, accuracy27.448111913687484% on100stories/21,874targets. Actual best.pt, best-accuracy.pt and latest.pt were preserved under `macm3:/Users/fernando/fly_wordbrain_connectorch/results/connectorch-decoder-v1-continuation/arms/F32rank128bounded/stopped-checkpoints`. Full CPU checkpoint audits verified manifests, parameter hashes, cursor, graph and selector identity. The reserved test was not scored.

The [accepted stop](../records/F32rank128bounded-accepted-early-stop.json), [signal receipt](../records/F32rank128bounded-stop-receipt.json) and [decision](../records/F32rank128bounded-early-stop-decision.json) retain exact hashes. Native signal-failure records remain historical evidence. E's preserved rank128 fixed-edge run is G's comparator; F is not queued for resumption.
