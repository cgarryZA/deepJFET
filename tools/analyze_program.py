#!/usr/bin/env python3
"""Analyze a 4004 program to determine which hardware resources it uses.

Takes a binary ROM image or hex file and produces a resource profile
that tells the build system which circuit variants to include.

Usage:
    python tools/analyze_program.py program.bin
    python tools/analyze_program.py program.hex --format ihex
    python tools/analyze_program.py program.bin --output profile.json
    python tools/analyze_program.py program.bin --verbose

Output (ResourceProfile):
    - registers_used:     set of register numbers (0-15) actually accessed
    - register_pairs_used: set of pair numbers (0-7)
    - max_stack_depth:    deepest subroutine nesting (0-3)
    - uses_ram:           whether RAM main memory is accessed
    - uses_ram_status:    whether RAM status chars are accessed
    - uses_ram_port:      whether RAM output port is used
    - uses_rom_port:      whether ROM I/O is used
    - uses_carry:         whether carry flag is used
    - rom_pages_used:     set of ROM page numbers (0-15) touched
    - alu_ops_used:       set of ALU operation mnemonics used
    - instruction_count:  total instructions decoded
    - scratchpad_min:     minimum scratchpad registers needed (nearest power of 2)
    - stack_depth_needed: actual stack levels needed
"""

import argparse
import json
import math
import os
import sys

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tools"))

from isa_4004 import decode, disassemble, Resource


class ResourceProfile:
    """Summary of hardware resources used by a program."""

    def __init__(self):
        self.registers_used = set()       # Individual register numbers 0-15
        self.register_pairs_used = set()  # Pair numbers 0-7
        self.max_stack_depth = 0
        self.uses_ram = False
        self.uses_ram_status = False
        self.uses_ram_port = False
        self.uses_rom_port = False
        self.uses_carry = False
        self.uses_accumulator = False
        self.uses_cmd_reg = False
        self.rom_pages_used = set()
        self.alu_ops_used = set()         # Mnemonics: ADD, SUB, IAC, DAC, etc.
        self.instruction_count = 0
        self.status_chars_used = set()    # 0-3
        self.branch_targets = set()

    @property
    def scratchpad_regs_needed(self) -> int:
        """Minimum scratchpad registers needed (round up to available variant).

        The 4004 has 16 registers. Common variants: 2, 4, 8, 16.
        """
        if not self.registers_used and not self.register_pairs_used:
            return 0

        # Expand pairs to individual registers
        all_regs = set(self.registers_used)
        for pair in self.register_pairs_used:
            all_regs.add(pair * 2)
            all_regs.add(pair * 2 + 1)

        if not all_regs:
            return 0

        # Need at least enough to cover the highest register number + 1
        max_reg = max(all_regs) + 1

        # Round up to nearest available variant size
        for size in [2, 4, 6, 8, 10, 12, 14, 16]:
            if size >= max_reg:
                return size
        return 16

    @property
    def stack_depth_needed(self) -> int:
        """Stack levels needed (0 if no subroutine calls, else 1-3)."""
        return min(self.max_stack_depth, 3)

    @property
    def stack_levels_needed(self) -> list:
        """List of stack level indices needed, e.g. [0], [0,1], [0,1,2]."""
        return list(range(self.stack_depth_needed))

    @property
    def rom_size_needed(self) -> int:
        """Number of ROM pages needed."""
        if not self.rom_pages_used:
            return 1
        return max(self.rom_pages_used) + 1

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "registers_used": sorted(self.registers_used),
            "register_pairs_used": sorted(self.register_pairs_used),
            "scratchpad_regs_needed": self.scratchpad_regs_needed,
            "max_stack_depth": self.max_stack_depth,
            "stack_depth_needed": self.stack_depth_needed,
            "stack_levels_needed": self.stack_levels_needed,
            "uses_ram": self.uses_ram,
            "uses_ram_status": self.uses_ram_status,
            "uses_ram_port": self.uses_ram_port,
            "uses_rom_port": self.uses_rom_port,
            "uses_carry": self.uses_carry,
            "uses_accumulator": self.uses_accumulator,
            "uses_cmd_reg": self.uses_cmd_reg,
            "rom_pages_used": sorted(self.rom_pages_used),
            "rom_size_needed": self.rom_size_needed,
            "alu_ops_used": sorted(self.alu_ops_used),
            "status_chars_used": sorted(self.status_chars_used),
            "instruction_count": self.instruction_count,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = ["=== Resource Profile ===", ""]

        # Registers
        all_regs = set(self.registers_used)
        for pair in self.register_pairs_used:
            all_regs.add(pair * 2)
            all_regs.add(pair * 2 + 1)
        lines.append(f"Scratchpad: {self.scratchpad_regs_needed} registers needed "
                      f"(uses: {sorted(all_regs) if all_regs else 'none'})")
        if self.register_pairs_used:
            lines.append(f"  Pairs: {', '.join(f'RP{p}(R{p*2},R{p*2+1})' for p in sorted(self.register_pairs_used))}")

        # Stack
        lines.append(f"Stack: {self.stack_depth_needed} levels needed "
                      f"(max depth observed: {self.max_stack_depth})")

        # ALU
        if self.alu_ops_used:
            lines.append(f"ALU ops: {', '.join(sorted(self.alu_ops_used))}")
        lines.append(f"Accumulator: {'yes' if self.uses_accumulator else 'no'}")
        lines.append(f"Carry flag: {'yes' if self.uses_carry else 'no'}")

        # Memory/IO
        lines.append(f"RAM: {'yes' if self.uses_ram else 'no'}")
        if self.uses_ram_status:
            lines.append(f"RAM status chars: {sorted(self.status_chars_used)}")
        lines.append(f"RAM port: {'yes' if self.uses_ram_port else 'no'}")
        lines.append(f"ROM port: {'yes' if self.uses_rom_port else 'no'}")
        if self.uses_cmd_reg:
            lines.append(f"Bank select (DCL): yes")

        # ROM
        lines.append(f"ROM pages: {sorted(self.rom_pages_used)} "
                      f"({self.rom_size_needed} pages needed)")
        lines.append(f"Instructions: {self.instruction_count}")

        # Build recommendations
        lines.append("")
        lines.append("=== Build Recommendations ===")
        lines.append(f"  scratchpad: {self.scratchpad_regs_needed}_reg variant")
        lines.append(f"  stack: {self.stack_depth_needed}_level variant")
        if not self.uses_ram and not self.uses_ram_status:
            lines.append("  ram: NOT NEEDED (can omit)")
        if not self.uses_rom_port:
            lines.append("  rom_port: NOT NEEDED (can omit)")

        return "\n".join(lines)


def analyze(data: bytes, base_address: int = 0) -> ResourceProfile:
    """Analyze a 4004 program and return its resource profile.

    Args:
        data: raw bytes of 4004 machine code
        base_address: starting ROM address

    Returns:
        ResourceProfile with all resource usage identified.
    """
    instructions = decode(data, base_address)
    profile = ResourceProfile()
    profile.instruction_count = len(instructions)

    # Track stack depth via simple simulation
    # (conservative: assume worst-case for conditional branches)
    current_depth = 0

    for inst in instructions:
        # ROM page tracking
        page = inst.address >> 8
        profile.rom_pages_used.add(page)

        # Register tracking
        if inst.register is not None:
            profile.registers_used.add(inst.register)

        if inst.register_pair is not None:
            profile.register_pairs_used.add(inst.register_pair)
            # FIN always reads pair 0 for the address
            if inst.mnemonic == "FIN":
                profile.register_pairs_used.add(0)

        # Stack tracking
        if inst.stack_push:
            current_depth += 1
            profile.max_stack_depth = max(profile.max_stack_depth, current_depth)
        if inst.stack_pop:
            current_depth = max(0, current_depth - 1)

        # Resource flags
        if Resource.ACC in inst.reads or Resource.ACC in inst.writes:
            profile.uses_accumulator = True
        if Resource.CY in inst.reads or Resource.CY in inst.writes:
            profile.uses_carry = True
        if Resource.RAM in inst.reads or Resource.RAM in inst.writes:
            profile.uses_ram = True
        if Resource.RAM_STATUS in inst.reads or Resource.RAM_STATUS in inst.writes:
            profile.uses_ram_status = True
            if inst.status_char is not None:
                profile.status_chars_used.add(inst.status_char)
        if Resource.RAM_PORT in inst.reads or Resource.RAM_PORT in inst.writes:
            profile.uses_ram_port = True
        if Resource.ROM_PORT in inst.reads or Resource.ROM_PORT in inst.writes:
            profile.uses_rom_port = True
        if Resource.CMD_REG in inst.writes:
            profile.uses_cmd_reg = True

        # ALU operation tracking
        if inst.mnemonic in ("ADD", "SUB", "ADM", "SBM", "IAC", "DAC",
                              "DAA", "TCS", "KBP", "CMA", "RAL", "RAR"):
            profile.alu_ops_used.add(inst.mnemonic)

        # Branch targets -> more ROM pages
        if inst.branch_target is not None:
            profile.branch_targets.add(inst.branch_target)
            target_page = inst.branch_target >> 8
            profile.rom_pages_used.add(target_page)

    return profile


def load_binary(filepath: str) -> bytes:
    """Load a raw binary file."""
    with open(filepath, "rb") as f:
        return f.read()


def load_ihex(filepath: str) -> bytes:
    """Load an Intel HEX file into a byte array."""
    data = bytearray()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith(":"):
                continue
            byte_count = int(line[1:3], 16)
            address = int(line[3:7], 16)
            record_type = int(line[7:9], 16)
            if record_type == 0x00:  # Data record
                # Extend data array if needed
                while len(data) < address + byte_count:
                    data.append(0x00)
                for j in range(byte_count):
                    data[address + j] = int(line[9 + j*2:11 + j*2], 16)
            elif record_type == 0x01:  # EOF
                break
    return bytes(data)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a 4004 program to determine hardware resource usage"
    )
    parser.add_argument("program", help="Path to program file (.bin or .hex)")
    parser.add_argument("--format", "-f", choices=["bin", "ihex", "hex"],
                        default="bin", help="Input format (default: bin)")
    parser.add_argument("--output", "-o", default=None,
                        help="Write JSON profile to file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full disassembly")

    args = parser.parse_args()

    # Load program
    if args.format in ("ihex", "hex"):
        data = load_ihex(args.program)
    else:
        data = load_binary(args.program)

    print(f"Loaded {len(data)} bytes from {args.program}")

    # Decode and optionally disassemble
    if args.verbose:
        instructions = decode(data)
        print(f"\n{disassemble(instructions)}\n")

    # Analyze
    profile = analyze(data)
    print(f"\n{profile.summary()}")

    # Write JSON output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        print(f"\nProfile written to {args.output}")

    return profile


if __name__ == "__main__":
    main()
