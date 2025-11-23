# Automations in Hiver

Automations help you streamline workflows by defining triggers and actions.

## Configure Automations

1. Go to `Settings > Automations`.
2. Create a new automation and provide a name.
3. Select a trigger (e.g., `new email`, `subject contains`, `tag applied`).
4. Add conditions to narrow scope (e.g., mailbox, sender domain).
5. Choose actions (e.g., `assign`, `apply tag`, `set status`, `add note`).
6. Save and enable the automation.

## Common Tips

- Prefer explicit conditions (subject keywords, mailbox) over broad triggers.
- Test with a staging mailbox before enabling in production.
- Use audit logs to debug when an automation doesn’t fire.

## Troubleshooting

- If automations duplicate tasks, check overlapping automations that share triggers.
- If delays occur, verify automation queue health and recent edits.
- After editing an automation, re-test to confirm expected behavior.