"""Allow ``python -m parlliament`` to invoke the command-line workflow."""

from .cli import main

raise SystemExit(main())
