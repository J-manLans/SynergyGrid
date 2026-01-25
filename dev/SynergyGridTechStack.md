## Core:
- **Python 3.9+** (type hints, dataclasses)
- **Gymnasium** — RL environment interface
- **NumPy** — numerical operations

**Visualization & Analysis:**
- **Matplotlib** — plotting benchmark results
- **Pygame** — optional live grid visualization

## Development:
- **Dataclasses** — clean resource/config definitions for data-holding classes
- **Type hints** — throughout codebase
- **pytest** — testing

## Optional (for later):
- **Pandas** — advanced data analysis/export

## **Examples**:

### Type Hints

```python
# Without type hints
def add_resource(x, y):
    return x + y

# With type hints (much clearer!)
def add_resource(x: int, y: int) -> int:
    return x + y

# For variables too
lives: int = 20
resources: list[str] = ["green", "red"]
```

### Dataclasses

```python
from dataclasses import dataclass

# Without dataclass
class Resource:
    def __init__(self, name, value, duration):
        self.name = name
        self.value = value
        self.duration = duration

    def __repr__(self):
        return f"Resource({self.name}, {self.value}, {self.duration})"

# With dataclass (same thing, auto-generated!)
@dataclass
class Resource:
    name: str
    value: int
    duration: int
```

### pytest

```python
# test_engine.py
import pytest
from core.engine import SynergyGridEnv

def test_agent_starts_with_20_lives():
    env = SynergyGridEnv()
    obs, info = env.reset()
    assert info['lives'] == 20

def test_moving_costs_one_life():
    env = SynergyGridEnv()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)  # Move up
    assert info['lives'] == 19
```