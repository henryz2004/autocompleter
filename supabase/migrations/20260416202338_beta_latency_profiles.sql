alter table public.beta_invocations
    add column if not exists requested_route text,
    add column if not exists profile_json jsonb;

alter table public.beta_proxy_requests
    add column if not exists profile_json jsonb;

alter table public.beta_proxy_attempts
    add column if not exists profile_json jsonb;

alter table public.beta_telemetry_events
    add column if not exists requested_route text,
    add column if not exists profile_json jsonb;
