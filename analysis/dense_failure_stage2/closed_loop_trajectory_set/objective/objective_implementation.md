# Exact trajectory-set objective

For each UID, every unique prefix state is forwarded once. Each route log-probability is the sum of its selected action log-probabilities. The loss is `-(logsumexp(route_logp)-log K)/T`. All K routes remain in one autograd graph; route sampling and hard responsibilities are disabled.
