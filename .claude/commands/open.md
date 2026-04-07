Open a CPU component schematic in LTSpice.

The user will specify a component name (e.g. "alu", "program_counter", "step_counter") and optionally a CPU name.

Run `python tools/open_ltspice.py <component>` to find and open the .asc file in LTSpice.

If no .asc file exists yet, tell the user the expected path where they should create one (inside cpus/<cpu>/<component>/<component>.asc).

If the user says something like "open the ALU" or "edit the program counter", extract the component name and run the tool.

Arguments: $ARGUMENTS
- Parse the arguments to extract the component name and optional --cpu flag
- Example: `/open alu` or `/open program_counter --cpu 4004`

Run the command:
```
python tools/open_ltspice.py $ARGUMENTS
```

Report back what happened — whether the file was opened, or if no .asc was found and where to create one.
