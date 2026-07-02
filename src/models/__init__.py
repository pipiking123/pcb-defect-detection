from .coord_attention import CoordAtt

import ultralytics.nn.tasks as _tasks

_tasks.CoordAtt = CoordAtt
