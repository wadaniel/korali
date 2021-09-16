git clone git@gitlab.ethz.ch:mavt-cse/modelBasedRL.git
mkdir Utils/
mkdir _model_cartpole/
cd modelBasedRL
cp Cartpole/model.py ../_model_cartpole/model.py
cp Cartpole/env.py ../_model_cartpole/env.py
cp Cartpole/cartpole.py ../_model_cartpole/cartpole.py
cp Cartpole/grid_search.py ../Utils/grid_search.py
cp Cartpole/group_results_onlyreal.py ../Utils/group_results_onlyreal.py
cp Cartpole/group_results.py ../Utils/group_results.py
cp Cartpole/plot_best_surrogate_based_model.py ../Utils/plot_best_surrogate_based_model.py
cp Cartpole/plot_results.py ../Utils/plot_results.py
cp Cartpole/plot_updates.py ../Utils/plot_updates.py
cp Cartpole/plot_optimize_3d.py ../Utils/plot_optimize_3d.py
cp Cartpole/render_cartpole.py ../Utils/render_cartpole.py
mkdir ../Results/
mkdir ../Visualization/
cd ..
rm -rf ./modelBasedRL
