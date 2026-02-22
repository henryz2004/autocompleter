# Commands

## Setup

```bash
source venv/bin/activate
```

## Clear the database

```bash
rm -f ~/.autocompleter/context.db
```

## Run autocompleter

```bash
python -m autocompleter
```

With logging:

```bash
python -m autocompleter --log-file /tmp/autocompleter.log --log-level DEBUG
```

## Dump AX tree

Press Ctrl+Space while focused on the target app, then Ctrl+C to quit.

```bash
python dump_ax_tree.py
```

To file:

```bash
python dump_ax_tree.py -o /tmp/ax_dump.log
```

## Dump context pipeline

Press Ctrl+Space while focused on the target app, then Ctrl+C to quit.

```bash
python dump_pipeline.py
```

To file:

```bash
python dump_pipeline.py -o /tmp/pipeline.log
```

Both modes:

```bash
python dump_pipeline.py --both-modes
```

## Run tests

```bash
./venv/bin/python -m pytest tests/ -v
```
