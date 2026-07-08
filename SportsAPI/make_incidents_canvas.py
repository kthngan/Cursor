#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


SOURCE = Path(__file__).resolve().parent.parent / "Data" / "SportsAPI" / "event_6571587_incidents_reconstructed.csv"
CANVAS = Path(
    r"C:\Users\user\.cursor\projects\c-Users-user-Documents-Cursor\canvases"
) / "tennis-incidents-reconstruction.canvas.tsx"


def main() -> int:
    rows = []
    with SOURCE.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            rows.append(
                {
                    key: row[key]
                    for key in [
                        "seq",
                        "utc_time",
                        "event_status_name",
                        "incident_name",
                        "participant_side",
                        "participant_name",
                        "point_to",
                        "game_score_after",
                        "sets_after",
                    ]
                }
            )

    code = f'''import {{ Button, H1, H2, Row, Stack, Text, Table, Grid, Stat, useCanvasState }} from "cursor/canvas";

const incidentRows = {json.dumps(rows, ensure_ascii=False, indent=2)};
const PAGE_SIZE = 50;

export default function TennisIncidentsReconstruction() {{
  const [page, setPage] = useCanvasState("page", 0);
  const totalPages = Math.ceil(incidentRows.length / PAGE_SIZE);
  const currentPage = Math.min(Math.max(page, 0), totalPages - 1);
  const start = currentPage * PAGE_SIZE;
  const visibleRows = incidentRows.slice(start, start + PAGE_SIZE);
  const tableRows = visibleRows.map((row) => [
    row.seq,
    row.utc_time,
    row.event_status_name,
    row.incident_name,
    row.participant_side,
    row.participant_name,
    row.point_to,
    row.game_score_after,
    row.sets_after,
  ]);

  return (
    <Stack gap={{18}} style={{{{ padding: 20 }}}}>
      <Stack gap={{6}}>
        <H1>Reconstructed Tennis Incident Timeline</H1>
        <Text tone="secondary">
          Lois Boisson vs Elena Rybakina, Wimbledon (Women) 2026, event_id 6571587.
          Source: StatScore events.show events_incidents, ordered by API update timestamp.
        </Text>
      </Stack>

      <Grid columns={{3}} gap={{12}}>
        <Stat label="Rows reconstructed" value={{String(incidentRows.length)}} />
        <Stat label="Final sets, Boisson-Rybakina" value="1-2" />
        <Stat label="Final games by set" value="4-6, 6-1, 3-6" />
      </Grid>

      <Stack gap={{8}}>
        <H2>All Reconstructed Incidents</H2>
        <Text tone="secondary" size="small">
          Game score and set score are derived while walking the incident stream. Tennis clock time is intentionally omitted.
        </Text>
        <Row gap={{8}} align="center">
          <Button variant="secondary" disabled={{currentPage === 0}} onClick={{() => setPage(currentPage - 1)}}>Previous</Button>
          <Text size="small" tone="secondary">
            Page {{currentPage + 1}} of {{totalPages}} · rows {{start + 1}}-{{Math.min(start + PAGE_SIZE, incidentRows.length)}} of {{incidentRows.length}}
          </Text>
          <Button variant="secondary" disabled={{currentPage >= totalPages - 1}} onClick={{() => setPage(currentPage + 1)}}>Next</Button>
        </Row>
        <Table
          headers={{["Seq", "UTC Time", "Status", "Incident", "Side", "Player", "Point", "Game Score After", "Sets After"]}}
          rows={{tableRows}}
          columnAlign={{["right", "left", "left", "left", "center", "left", "center", "center", "center"]}}
          striped
          stickyHeader
        />
      </Stack>
    </Stack>
  );
}}
'''
    CANVAS.write_text(code, encoding="utf-8")
    print(CANVAS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
