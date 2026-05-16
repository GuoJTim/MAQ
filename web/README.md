# MAQ Experiment Dashboard

Run the Flask service from the repository root:

```bash
pip install -r web/requirements.txt
python web/server.py --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000/
```

The default CSV is served from:

```text
evaluation_results/agent_performance_door-human-v1.csv
```

To use a different CSV in `evaluation_results/`:

```bash
python web/server.py --csv your_file.csv
```
