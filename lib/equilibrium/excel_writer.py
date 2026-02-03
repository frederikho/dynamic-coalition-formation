"""
Custom Excel writer for strategy tables that exactly matches the original format.

This creates Excel files from scratch to match the exact structure, styling,
and formatting of the original strategy tables.
"""

from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import pandas as pd
import numpy as np


# Color scheme (from original files)
COLORS = {
    'proposer_header': 'FF4D9CC9',      # Blue for "Proposer X"
    'state_name': 'FFEEBF99',           # Beige/orange for state names in column A
    'proposition_row': 'FF99C7E0',      # Light blue for proposition rows
    'acceptance_row': 'FFCCECE3',       # Very light blue/green for acceptance rows
    'player_name': 'FF66C5AB',          # Green for player names in acceptance rows
    'self_acceptance': 'FFE28E4D',      # Orange for player accepting own proposal
    'nan_cell': 'FFD9D9D9',             # Gray for NaN/None cells
}


def write_strategy_table_excel(df: pd.DataFrame, excel_file_path: str, players: list,
                              effectivity: dict = None, states: list = None):
    """
    Write a strategy DataFrame to Excel with exact formatting matching original files.

    Args:
        df: Strategy DataFrame with MultiIndex structure:
            - Index: (Current State, Type, Player)
            - Columns: (Proposer, Next State)
        excel_file_path: Output file path
        players: List of player names (e.g., ['W', 'T', 'C'])
        effectivity: Effectivity correspondence dict (optional)
        states: List of state names (optional, inferred from df if not provided)
    """
    # Create new workbook
    wb = Workbook()
    ws = wb.active

    # Extract structure from DataFrame
    if states is None:
        states = df.index.get_level_values(0).unique().tolist()
    proposers = df.columns.get_level_values(0).unique().tolist()

    # Helper function to check if transition is unilateral
    def is_unilateral_transition(current_state, next_state, proposer_label, effectivity_dict):
        """
        Check if a transition can be done without consulting others.

        Unilateral transitions:
        1. Maintaining status quo (current_state == next_state)
        2. Leaving a coalition (if only proposer or no one needs to approve)
        """
        # Extract proposer name from label (e.g., "Proposer W" -> "W")
        proposer_name = proposer_label.split()[-1] if 'Proposer' in proposer_label else proposer_label

        # Same state is always unilateral (maintaining status quo)
        if current_state == next_state:
            return True

        # If we don't have effectivity info, can't determine
        if effectivity_dict is None:
            return False

        # Check approval committee for this transition
        # Effectivity key: (proposer_name, current_state, next_state, responder)
        # If responder in committee, effectivity[(proposer_name, current_state, next_state, responder)] == 1
        approval_committee = []
        for player in players:
            key = (proposer_name, current_state, next_state, player)
            if key in effectivity_dict and effectivity_dict[key] == 1:
                approval_committee.append(player)

        # Unilateral if:
        # - Empty approval committee, OR
        # - Only proposer in approval committee
        is_unilateral = (len(approval_committee) == 0 or
                        (len(approval_committee) == 1 and proposer_name in approval_committee))

        return is_unilateral

    # Set column widths
    ws.column_dimensions['A'].width = 5.109375
    ws.column_dimensions['B'].width = 8.77734375
    ws.column_dimensions['C'].width = 3.77734375

    # Data columns: uniform width of 7.0 (approximately 1.85 cm)
    for col_idx in range(4, 4 + len(proposers) * len(states)):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = 7.0

    # Set row heights
    ws.row_dimensions[1].height = 16.5
    ws.row_dimensions[2].height = 14.85
    ws.row_dimensions[3].height = 14.85

    # Write headers
    # Row 1: Proposer headers (merged cells)
    col = 4
    for proposer_label in proposers:
        # Merge cells for this proposer (spans len(states) columns)
        start_col = col
        end_col = col + len(states) - 1
        ws.merge_cells(start_row=1, start_column=start_col, end_row=1, end_column=end_col)

        # Write proposer name in merged cell
        ws.cell(row=1, column=start_col, value=proposer_label)
        ws.cell(row=1, column=start_col).fill = PatternFill(
            start_color=COLORS['proposer_header'],
            end_color=COLORS['proposer_header'],
            fill_type='solid'
        )
        ws.cell(row=1, column=start_col).font = Font(name='Calibri', bold=True, size=9)
        ws.cell(row=1, column=start_col).alignment = Alignment(horizontal='center', vertical='center')

        col = end_col + 1

    # Row 2: State headers (repeated for each proposer)
    col = 4
    for proposer_label in proposers:
        for state in states:
            ws.cell(row=2, column=col, value=state)
            ws.cell(row=2, column=col).font = Font(name='Calibri', size=9)
            ws.cell(row=2, column=col).alignment = Alignment(horizontal='center', vertical='center')
            col += 1

    # Write data rows
    current_row = 3
    for state in states:
        # Proposition row
        ws.cell(row=current_row, column=1, value=state)
        ws.cell(row=current_row, column=1).fill = PatternFill(
            start_color=COLORS['state_name'],
            end_color=COLORS['state_name'],
            fill_type='solid'
        )
        ws.cell(row=current_row, column=1).font = Font(name='Calibri', bold=True, size=9)
        ws.cell(row=current_row, column=1).alignment = Alignment(horizontal='right', vertical='center')

        ws.cell(row=current_row, column=2, value='Proposition')
        ws.cell(row=current_row, column=2).fill = PatternFill(
            start_color=COLORS['proposition_row'],
            end_color=COLORS['proposition_row'],
            fill_type='solid'
        )
        ws.cell(row=current_row, column=2).font = Font(name='Calibri', size=9)
        ws.cell(row=current_row, column=2).alignment = Alignment(horizontal='center', vertical='center')

        # Column 3: Write string 'NaN' to prevent pandas forward-fill when reading
        ws.cell(row=current_row, column=3, value='NaN')
        ws.cell(row=current_row, column=3).fill = PatternFill(
            start_color=COLORS['proposition_row'],
            end_color=COLORS['proposition_row'],
            fill_type='solid'
        )
        ws.cell(row=current_row, column=3).font = Font(name='Calibri', size=9)
        ws.cell(row=current_row, column=3).alignment = Alignment(horizontal='center', vertical='center')

        # Fill proposition probabilities
        col = 4
        for proposer_label in proposers:
            for next_state in states:
                val = df.loc[(state, 'Proposition', np.nan), (proposer_label, next_state)]
                # Only show values != 0
                if pd.isna(val) or val == 0:
                    ws.cell(row=current_row, column=col, value=None)
                else:
                    ws.cell(row=current_row, column=col, value=float(val))

                ws.cell(row=current_row, column=col).fill = PatternFill(
                    start_color=COLORS['proposition_row'],
                    end_color=COLORS['proposition_row'],
                    fill_type='solid'
                )
                ws.cell(row=current_row, column=col).font = Font(name='Calibri', size=9)
                ws.cell(row=current_row, column=col).alignment = Alignment(horizontal='center', vertical='center')
                col += 1

        current_row += 1

        # Acceptance rows (one per player)
        for player in players:
            # State name (all acceptance rows)
            ws.cell(row=current_row, column=1, value=state)
            ws.cell(row=current_row, column=1).fill = PatternFill(
                start_color=COLORS['state_name'],
                end_color=COLORS['state_name'],
                fill_type='solid'
            )
            ws.cell(row=current_row, column=1).font = Font(name='Calibri', bold=True, size=9)
            ws.cell(row=current_row, column=1).alignment = Alignment(horizontal='right', vertical='center')

            # Type (all acceptance rows)
            ws.cell(row=current_row, column=2, value='Acceptance')
            ws.cell(row=current_row, column=2).fill = PatternFill(
                start_color=COLORS['acceptance_row'],
                end_color=COLORS['acceptance_row'],
                fill_type='solid'
            )
            ws.cell(row=current_row, column=2).font = Font(name='Calibri', size=9)
            ws.cell(row=current_row, column=2).alignment = Alignment(horizontal='center', vertical='center')

            # Player name
            ws.cell(row=current_row, column=3, value=player)
            ws.cell(row=current_row, column=3).fill = PatternFill(
                start_color=COLORS['player_name'],
                end_color=COLORS['player_name'],
                fill_type='solid'
            )
            ws.cell(row=current_row, column=3).font = Font(name='Calibri', size=9)
            ws.cell(row=current_row, column=3).alignment = Alignment(horizontal='center', vertical='center')

            # Fill acceptance probabilities
            col = 4
            for proposer_label in proposers:
                proposer_name = proposer_label.split()[-1]  # Extract "W" from "Proposer W"

                for next_state in states:
                    val = df.loc[(state, 'Acceptance', player), (proposer_label, next_state)]

                    if pd.isna(val):
                        # Grey: Not in approval committee
                        ws.cell(row=current_row, column=col, value=None)
                        ws.cell(row=current_row, column=col).fill = PatternFill(
                            start_color=COLORS['nan_cell'],
                            end_color=COLORS['nan_cell'],
                            fill_type='solid'
                        )
                    else:
                        # Has value: Player is in approval committee
                        ws.cell(row=current_row, column=col, value=float(val))

                        # Dark orange if: player == proposer AND unilateral transition
                        # (transitions the proposer can do without consulting others)
                        is_unilateral = is_unilateral_transition(state, next_state, proposer_label, effectivity)
                        is_self = (player == proposer_name)

                        if is_self and is_unilateral:
                            # Dark orange: Player accepting their own unilateral action
                            ws.cell(row=current_row, column=col).fill = PatternFill(
                                start_color=COLORS['self_acceptance'],  # FFE28E4D
                                end_color=COLORS['self_acceptance'],
                                fill_type='solid'
                            )
                        else:
                            # Light green: In approval committee, but not self-unilateral
                            ws.cell(row=current_row, column=col).fill = PatternFill(
                                start_color=COLORS['acceptance_row'],  # FFCCECE3
                                end_color=COLORS['acceptance_row'],
                                fill_type='solid'
                            )

                    ws.cell(row=current_row, column=col).font = Font(name='Calibri', size=9)
                    ws.cell(row=current_row, column=col).alignment = Alignment(horizontal='center', vertical='center')

                    col += 1

            current_row += 1

    # Add borders
    # Define border styles
    thin_border = Side(style='thin', color='000000')

    # Determine table dimensions
    last_row = current_row - 1
    last_col = 3 + len(proposers) * len(states)  # 3 label cols + data cols

    # Calculate state group boundaries (horizontal borders between states)
    # Each state has 1 proposition row + len(players) acceptance rows
    rows_per_state = 1 + len(players)
    state_boundaries_rows = []
    for state_idx in range(len(states) - 1):  # Don't add border after last state
        boundary_row = 2 + (state_idx + 1) * rows_per_state  # 2 header rows + state rows
        state_boundaries_rows.append(boundary_row)

    # Apply borders
    for row in range(1, last_row + 1):
        for col in range(1, last_col + 1):
            cell = ws.cell(row=row, column=col)
            border_kwargs = {}

            # Outside border (medium)
            if row == 1:
                border_kwargs['top'] = thin_border
            if row == last_row:
                border_kwargs['bottom'] = thin_border
            if col == 1:
                border_kwargs['left'] = thin_border
            if col == last_col:
                border_kwargs['right'] = thin_border

            # Vertical border between label columns (A-C) and data columns (D onwards)
            if col == 3:
                border_kwargs['right'] = thin_border
            if col == 4:
                border_kwargs['left'] = thin_border

            # Vertical borders between proposers (after columns H, M, etc.)
            proposer_boundaries = [3 + len(states) * (i + 1) for i in range(len(proposers) - 1)]
            if col in proposer_boundaries:
                border_kwargs['right'] = thin_border
            if col - 1 in proposer_boundaries and col > 3:
                border_kwargs['left'] = thin_border

            # Horizontal border between header rows (1-2) and data rows (3+)
            if row == 2:
                border_kwargs['bottom'] = thin_border
            if row == 3:
                border_kwargs['top'] = thin_border

            # Horizontal borders between states
            if row in state_boundaries_rows:
                border_kwargs['bottom'] = thin_border
            if row - 1 in state_boundaries_rows:
                border_kwargs['top'] = thin_border

            # Apply border if any
            if border_kwargs:
                # Preserve existing borders (take the stronger one if both exist)
                existing_border = cell.border
                final_border = Border(
                    left=border_kwargs.get('left', existing_border.left),
                    right=border_kwargs.get('right', existing_border.right),
                    top=border_kwargs.get('top', existing_border.top),
                    bottom=border_kwargs.get('bottom', existing_border.bottom)
                )
                cell.border = final_border

    # Save workbook
    wb.save(excel_file_path)


if __name__ == '__main__':
    # Example usage
    import sys
    if len(sys.argv) > 1:
        # Read existing DataFrame and rewrite
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else input_file
        players = sys.argv[3].split(',') if len(sys.argv) > 3 else ['W', 'T', 'C']

        df = pd.read_excel(input_file, header=[0, 1], index_col=[0, 1, 2])
        write_strategy_table_excel(df, output_file, players)
        print(f"Rewrote {input_file} to {output_file}")
