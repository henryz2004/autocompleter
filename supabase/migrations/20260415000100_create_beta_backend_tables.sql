create table if not exists public.beta_installs (
    install_id text primary key,
    key_hash text not null unique,
    status text not null,
    label text,
    created_at timestamptz not null default timezone('utc', now()),
    revoked_at timestamptz,
    notes text
);

create table if not exists public.beta_proxy_requests (
    request_id text primary key,
    install_id text not null references public.beta_installs (install_id),
    requested_model text,
    resolved_model text,
    primary_upstream text,
    fallback_used boolean not null default false,
    stream boolean not null default false,
    status text not null,
    error_type text,
    latency_ms integer,
    message_count integer,
    input_chars_estimate integer,
    output_chars_estimate integer,
    created_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.beta_telemetry_events (
    event_id text primary key,
    install_id text not null references public.beta_installs (install_id),
    event_name text not null,
    payload_json jsonb not null,
    received_at timestamptz not null default timezone('utc', now())
);

create index if not exists beta_installs_key_hash_idx on public.beta_installs (key_hash);
create index if not exists beta_proxy_requests_install_id_idx on public.beta_proxy_requests (install_id, created_at desc);
create index if not exists beta_telemetry_events_install_id_idx on public.beta_telemetry_events (install_id, received_at desc);

