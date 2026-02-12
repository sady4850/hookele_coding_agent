---
name: Mailman
description: For configuring mailing list servers with Postfix and Mailman3, including subscription workflows and mail delivery.
---

## Mailman + Postfix Configuration

### Architecture (Bidirectional Flow)

**Inbound**: External mail → Postfix → LMTP → Mailman (processes subscriptions, posts)
**Outbound**: Mailman → Postfix → subscriber mailboxes (`/var/mail/<user>`)

Both directions must work. Common failure: configuring only inbound while neglecting outbound.

### Mandatory Verification (ALL THREE)

1. **Join**: Send to `list-join@domain`, confirm, verify membership
2. **Post/Broadcast**: Send to `list@domain`, verify ALL subscribers receive it
3. **Leave**: Send to `list-leave@domain`, confirm, verify removal

**Never declare success based on only subscription tests passing.**

### Subscription Policy Settings

For tasks requiring user confirmation on join/leave:
- `subscription_policy`: should require confirmation (not `open` without confirm)
- `unsubscription_policy`: must also require confirmation if task specifies it
- Check both policies explicitly - they are separate settings

### Common Pitfalls

1. **Domain conflicts**: If list domain is in Postfix's `mydestination`, it treats ALL addresses as local. Use `transport_maps` to override for list addresses.

2. **LMTP gaps**: Subscription requests may work while posts fail. Test actual post messages, not just subscriptions.

3. **Incomplete outbound**: Verify Mailman's "out" runner is active and messages reach subscriber mailboxes.

4. **Premature success**: A single test passing doesn't mean full functionality works.

### Debugging Checklist

1. Trace message path: enters system → reaches LMTP → Mailman processes → outbound delivery → subscriber mailbox
2. Check queues: `mailq`, Mailman queue directories
3. Check logs: `/var/log/mail.log`, `/var/log/mailman3/`
4. Verify LMTP port is listening

### Incremental Verification

1. Basic Postfix local mail works
2. Mailman installed and configured
3. LMTP connection (Postfix → Mailman)
4. Subscription flow (join AND leave with confirmation)
5. Broadcast flow (post reaches all subscribers)
6. Full integration test
