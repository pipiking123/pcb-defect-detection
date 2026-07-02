"""Training entry point.

`src.models` must be imported before any ultralytics import that triggers
YAML model parsing, since it registers custom modules (e.g. CoordAtt) into
the ultralytics.nn.tasks namespace that parse_model()'s eval() resolves
against. `src.losses` must be imported before any BboxLoss instance is
constructed, since it monkey-patches BboxLoss.__init__/forward for WIoU v3
when IOU_TYPE=wiou is set (no-op otherwise).
"""

import src.models  # noqa: F401  (registers CoordAtt into ultralytics.nn.tasks)
import src.losses  # noqa: F401  (patches BboxLoss for WIoU v3 if IOU_TYPE=wiou)

from ultralytics import YOLO
