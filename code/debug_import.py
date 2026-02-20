import sys
import os
print("Current working directory:", os.getcwd())
print("Sys path:", sys.path)

try:
    import models.DWCN
    print("Successfully imported models.DWCN")
    print("Attributes of models.DWCN:", dir(models.DWCN))
    from models.DWCN import DWCN
    print("Successfully imported DWCN class")
except ImportError as e:
    print("ImportError:", e)
except Exception as e:
    print("Other error:", e)
