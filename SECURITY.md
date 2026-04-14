# Security Policy

## Supported Scope

This project is early-stage and currently maintained on a best-effort basis.

Please report:

- vulnerabilities that expose local data unexpectedly
- prompt or dump flows that leak sensitive context more broadly than intended
- injection behavior that can write text into the wrong target
- unsafe defaults involving provider credentials, local files, or network requests

## Reporting

Please do not open a public issue for a suspected security problem.

Instead, contact the maintainer privately and include:

- a short description of the issue
- impact and affected versions or commit range
- reproduction steps
- whether sensitive data may have been exposed

If you do not yet have a dedicated security contact address before launch, add one before making the repository public.

## Sensitive Data Reminder

This app can process active-window context, drafts, URLs, terminal content, and other user-visible text. Security reports involving privacy or accidental exfiltration are in scope even if the issue is caused by prompt assembly, logging, or debugging tools such as `--dump-dir`.

