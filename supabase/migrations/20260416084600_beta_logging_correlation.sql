alter table public.beta_installs enable row level security;
alter table public.beta_proxy_requests enable row level security;
alter table public.beta_telemetry_events enable row level security;

create table if not exists public.beta_invocations (
    invocation_id text primary key,
    install_id text not null references public.beta_installs (install_id),
    trigger_type text not null check (
        trigger_type in ('manual', 'auto', 'regenerate', 'post_accept')
    ),
    mode text not null check (
        mode in ('continuation', 'reply')
    ),
    source_app text not null,
    app_category text check (
        app_category in ('terminal', 'browser', 'chat', 'editor', 'other')
    ),
    started_at timestamptz,
    first_displayed_at timestamptz,
    stream_completed_at timestamptz,
    resolved_at timestamptz,
    first_display_latency_ms integer,
    stream_complete_latency_ms integer,
    dwell_ms integer,
    suggestion_count integer,
    fallback_used boolean,
    final_outcome text check (
        final_outcome is null
        or final_outcome in (
            'accepted',
            'partial_accepted',
            'dismissed',
            'typed_through',
            'superseded',
            'no_suggestions',
            'error'
        )
    ),
    accepted_rank integer,
    accepted_length_bucket text,
    dismiss_reason text check (
        dismiss_reason is null
        or dismiss_reason in ('explicit', 'typing', 'superseded')
    ),
    error_type text,
    proxy_request_id text,
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now())
);

alter table public.beta_invocations enable row level security;

alter table public.beta_proxy_requests
    add column if not exists invocation_id text,
    add column if not exists attempt_count integer not null default 0,
    add column if not exists first_attempt_started_at timestamptz,
    add column if not exists completed_at timestamptz;

create table if not exists public.beta_proxy_attempts (
    attempt_id text primary key,
    request_id text not null,
    install_id text not null references public.beta_installs (install_id),
    invocation_id text,
    attempt_number integer not null,
    upstream_name text not null,
    resolved_model text,
    is_fallback_attempt boolean not null default false,
    stream boolean not null default false,
    status text not null,
    error_type text,
    latency_ms integer,
    input_chars_estimate integer,
    output_chars_estimate integer,
    started_at timestamptz not null,
    completed_at timestamptz not null,
    created_at timestamptz not null default timezone('utc', now())
);

alter table public.beta_proxy_attempts enable row level security;

alter table public.beta_telemetry_events
    add column if not exists invocation_id text,
    add column if not exists request_id text,
    add column if not exists event_time timestamptz,
    add column if not exists trigger_type text,
    add column if not exists mode text,
    add column if not exists source_app text,
    add column if not exists app_category text;

create index if not exists beta_invocations_install_id_started_at_idx
    on public.beta_invocations (install_id, started_at desc);
create index if not exists beta_invocations_source_app_started_at_idx
    on public.beta_invocations (source_app, started_at desc);
create index if not exists beta_invocations_outcome_started_at_idx
    on public.beta_invocations (final_outcome, started_at desc);

create index if not exists beta_proxy_requests_invocation_id_idx
    on public.beta_proxy_requests (install_id, invocation_id, created_at desc);
create index if not exists beta_proxy_attempts_request_id_idx
    on public.beta_proxy_attempts (request_id, attempt_number);
create index if not exists beta_proxy_attempts_invocation_id_idx
    on public.beta_proxy_attempts (install_id, invocation_id, started_at desc);

create index if not exists beta_telemetry_events_invocation_id_idx
    on public.beta_telemetry_events (install_id, invocation_id, event_time desc);
