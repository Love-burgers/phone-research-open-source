# Security Policy

This project runs a multi-turn phone selection assistant. Keep the default localhost binding for personal use. Publishing this repository does not publish a hosted service or the author's API credentials.

## Credentials and access

- Copy `.env.example` locally; never commit `.env`, API keys, cookies or access tokens. The example contains no credentials.
- A homepage API Key stays in the current tab's JavaScript memory. It is sent to this backend only with the chat request and overrides the configured chat key for that request. It is not saved in browser storage, conversation records or configuration. A rejected personal key does not fall back to the server key.
- Enter personal keys only over loopback HTTP or HTTPS. The provider endpoint must also use HTTPS or loopback. A remote deployment operator can still access requests: use a deployment you trust. Separately configured remote embeddings use the deployment's embedding credentials.
- Remote listening requires `CHAT_ACCESS_TOKEN`. This is a shared access gate, not a multi-user account system. Anyone who passes the gate and knows a session ID can access that session. Do not use it for confidential multi-user hosting without additional identity and authorization controls.
- Conversations and usage accounting are stored in SQLite under `runs/` by default. They contain user input, source excerpts and model messages. Private runs, real-conversation regression fixtures and historical review output are excluded from the public release.

## Existing boundaries and limitations

- Defaults limit each IP to six chat requests per minute, and the deployment to 100 chat turns, 200 model requests and 600000 tokens per day. Limits are configurable; zero disables the corresponding limit. Daily accounting resets at midnight UTC+8 and uses atomic SQLite reservations. These limits do not account for unrelated applications using the same provider account or guarantee a monetary bill.
- Tool calls, model calls, context size and per-turn duration are bounded. Deadline checks are cooperative, not a guarantee of immediate process interruption.
- Source URLs and redirects are checked for HTTP(S), credentials, allowed ports and non-public IP addresses. DNS checks are separate from HTTP connections; this is not complete DNS rebinding protection. Fetches also have timeout, redirect and decoded-content limits, not an absolute bound on decoder peak memory.
- Web excerpts are untrusted input. Prompt instructions and citation checks are not a verified prompt-injection defense or proof of semantic truth. Parameter evidence does not establish measured gaming or camera performance.
- Citation IDs and phone scope are checked before display; invalid claims use a bounded fallback. UI text is rendered with `textContent`, and citation links are restricted to credential-free HTTP(S) URLs.
- Concurrent conversation writes use revision checks. A conflict returns 409 without overwriting the saved conversation. Provider failures return fixed public messages rather than raw exception details.
- Do not bypass logins, CAPTCHAs, paywalls or access controls.

## Reporting

Report security issues privately to the repository owner. Include reproducible local steps without secrets or attacks against a live deployment.

