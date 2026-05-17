"""4002 RAM / 4001 ROM-port emulator state.

Models the full off-CPU address space that a 4004 program can touch via
the SRC / WRM / RDM / WR0..3 / RD0..3 / SBM / ADM / WMP / WRR / RDR /
DCL instruction group. The CPU itself is in CPU4004Sim
(tools/rom_emulator.py); this module is the "everything else the CPU
talks to over the data bus" companion.

Address space layout (per the MCS-4 ISA reference):

    bank        0..7    selected by DCL → drives CM-RAM0..3 lines
    chip        0..3    top 2 bits of the SRC address byte (high nibble)
    register    0..3    bottom 2 bits of the SRC address byte (high nibble)
    character   0..15   the SRC address byte's low nibble

  Per (bank,chip,register):
    16 × 4-bit main memory characters (M)
     4 × 4-bit status characters (MS0..MS3)
  Per (bank,chip):
     1 × 4-bit RAM output port (driven by WMP, latched until next WMP)

  Per (rom_chip = high nibble of SRC byte):
     1 × 4-bit ROM input pins (read by RDR — caller-configurable)
     1 × 4-bit ROM output port (driven by WRR, latched until next WRR)

The SRC address remains latched until the next SRC. DCL state remains
latched until the next DCL or RESET.
"""

from typing import Optional


class RAMState:
    """In-memory model of every RAM/ROM-port location a 4004 program can
    address. All values are 0..15 (4-bit nibbles)."""

    def __init__(self):
        # Main memory  [bank][chip][register][character]  → 0..15
        self.main = [[[[0] * 16 for _ in range(4)]
                      for _ in range(4)] for _ in range(8)]
        # Status chars [bank][chip][register][status_index]  → 0..15
        self.status = [[[[0] * 4 for _ in range(4)]
                        for _ in range(4)] for _ in range(8)]
        # RAM output port [bank][chip] → 0..15
        self.ram_port_out = [[0] * 4 for _ in range(8)]
        # ROM port pins [rom_chip] → 0..15
        self.rom_port_in = [0] * 16
        self.rom_port_out = [0] * 16

        # State latched by previous SRC
        self.chip = 0
        self.reg = 0
        self.char = 0
        # Top 4 bits of SRC byte also select a ROM chip (for WRR/RDR)
        self.rom_chip = 0

        # Bank selected by DCL — 3-bit value 0..7
        # (Default after RESET = bank 0, matching ISA "no DCL sent" behaviour)
        self.bank = 0

    # ── Address latching ────────────────────────────────────────────────

    def src_latch(self, high_nibble: int, low_nibble: int) -> None:
        """SRC sends 8 bits in two cycles. The high nibble's top 2 bits
        select the chip, bottom 2 bits select the register. The low
        nibble is the character index. The full high nibble also acts
        as the ROM chip selector for any subsequent WRR/RDR."""
        h = high_nibble & 0xF
        l = low_nibble & 0xF
        self.chip = (h >> 2) & 0x3
        self.reg = h & 0x3
        self.char = l
        self.rom_chip = h

    def dcl(self, acc: int) -> None:
        """DCL: the bottom 3 bits of ACC select the bank. (The ISA table
        is essentially one-hot per bit selecting CM-RAM1..3 plus the
        default CM-RAM0 when all three bits are zero — but for the
        purpose of this emulator we just treat it as a 0..7 bank index
        and let the schematic-level CM-RAM logic decide what fires.)"""
        self.bank = acc & 0x7

    # ── Main / status memory ────────────────────────────────────────────

    def read_main(self) -> int:
        return self.main[self.bank][self.chip][self.reg][self.char]

    def write_main(self, val: int) -> None:
        self.main[self.bank][self.chip][self.reg][self.char] = val & 0xF

    def read_status(self, n: int) -> int:
        return self.status[self.bank][self.chip][self.reg][n & 0x3]

    def write_status(self, n: int, val: int) -> None:
        self.status[self.bank][self.chip][self.reg][n & 0x3] = val & 0xF

    # ── Ports ───────────────────────────────────────────────────────────

    def write_ram_port(self, val: int) -> None:
        """WMP: write ACC to the 4-bit output port of the currently
        SRC-selected RAM chip (in the current bank)."""
        self.ram_port_out[self.bank][self.chip] = val & 0xF

    def write_rom_port(self, val: int) -> None:
        """WRR: write ACC to the currently SRC-selected ROM chip's port."""
        self.rom_port_out[self.rom_chip] = val & 0xF

    def read_rom_port(self) -> int:
        """RDR: read ACC from the currently SRC-selected ROM chip's port."""
        return self.rom_port_in[self.rom_chip]

    # ── Test / debug fixtures ──────────────────────────────────────────

    def preload_main(self, bank: int, chip: int, reg: int,
                     values: list[int]) -> None:
        """Initialise a register's 16 main-memory chars (for test
        programs that read from RAM without writing first)."""
        for i, v in enumerate(values[:16]):
            self.main[bank][chip][reg][i] = v & 0xF

    def set_rom_input(self, rom_chip: int, val: int) -> None:
        """Configure what RDR will return when chip `rom_chip` is
        SRC-selected. Default 0; programs that test RDR should set
        this before running."""
        self.rom_port_in[rom_chip & 0xF] = val & 0xF

    # ── Snapshotting ────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Return a compact summary of all non-zero state. Useful for
        building expected_state.yaml entries for RAM-using programs."""
        out = {'bank': self.bank, 'chip': self.chip,
               'reg': self.reg, 'char': self.char,
               'rom_chip': self.rom_chip}
        nz_main, nz_status = [], []
        nz_ram_port, nz_rom_port_out = [], []
        for b in range(8):
            for c in range(4):
                if self.ram_port_out[b][c] != 0:
                    nz_ram_port.append((b, c, self.ram_port_out[b][c]))
                for r in range(4):
                    for ch in range(16):
                        v = self.main[b][c][r][ch]
                        if v != 0:
                            nz_main.append((b, c, r, ch, v))
                    for s in range(4):
                        v = self.status[b][c][r][s]
                        if v != 0:
                            nz_status.append((b, c, r, s, v))
        for rc in range(16):
            if self.rom_port_out[rc] != 0:
                nz_rom_port_out.append((rc, self.rom_port_out[rc]))
        out['main_nonzero'] = nz_main
        out['status_nonzero'] = nz_status
        out['ram_port_nonzero'] = nz_ram_port
        out['rom_port_out_nonzero'] = nz_rom_port_out
        return out
