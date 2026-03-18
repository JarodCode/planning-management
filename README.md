## Virtual Environment Setup

### Create the virtual environment (first time only)
```bash
python3 -m venv myvenv
```

### Activate the virtual environment

On Linux/Mac:
```bash
source myvenv/bin/activate
```

On Windows:
```bash
myvenv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Deactivate when you're done
```bash
deactivate
```

> Your terminal should show `(myvenv)` at the beginning of the line when the virtual environment is active.