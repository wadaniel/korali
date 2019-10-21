#!/usr/bin/env python3

import os
import sys
import json
import plotly.figure_factory as ff

with open('_korali_result/final.json') as f:
 jsString = f.read()
js  = json.loads(jsString)

df = []
for x in js["Profiling"]:
 for s in js["Profiling"][x]["Segments"]:
  df.append( dict(Task=x, Start=s["Start Time"]*4096, Finish=s["End Time"]*4096, SampleId=s["Segment Id"]*2) )

fig = ff.create_gantt(df, group_tasks=True, index_col='SampleId')
fig.write_html('first_figure.html')
