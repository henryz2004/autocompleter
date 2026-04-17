create table if not exists public.beta_debug_artifacts (
    artifact_id text primary key,
    install_id text not null references public.beta_installs (install_id),
    invocation_id text,
    artifact_type text not null,
    source_app text,
    trigger_type text,
    payload_json jsonb not null,
    created_at timestamptz not null default timezone('utc', now())
);

alter table public.beta_debug_artifacts enable row level security;

create index if not exists beta_debug_artifacts_install_id_created_at_idx
    on public.beta_debug_artifacts (install_id, created_at desc);
create index if not exists beta_debug_artifacts_invocation_id_created_at_idx
    on public.beta_debug_artifacts (invocation_id, created_at desc);
create index if not exists beta_debug_artifacts_artifact_type_created_at_idx
    on public.beta_debug_artifacts (artifact_type, created_at desc);
create index if not exists beta_debug_artifacts_source_app_created_at_idx
    on public.beta_debug_artifacts (source_app, created_at desc);
