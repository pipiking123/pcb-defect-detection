"""Training entry point.

`src.models` must be imported before any ultralytics import that triggers
YAML model parsing, since it registers custom modules (e.g. CoordAtt) into
the ultralytics.nn.tasks namespace that parse_model()'s eval() resolves
against.
"""

import src.models  # noqa: F401  (registers CoordAtt into ultralytics.nn.tasks)

from ultralytics import YOLO
