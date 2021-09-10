git clone git@gitlab.ethz.ch:mavt-cse/modelBasedRL.git
mkdir Utils/
mkdir _modelCoord/
cd modelBasedRL
cp model.py ../_modelCoord/model.py
cp env.py ../_modelCoord/env.py
cp cartpole.py ../_modelCoord/cartpole.py
cp grid_search.py ../Utils/grid_search.py
cp group_results_onlyreal.py ../Utils/group_results_onlyreal.py
cp group_results.py ../Utils/group_results.py
cp plot_best_surrogate_based_model.py ../Utils/plot_best_surrogate_based_model.py
cp plot_results.py ../Utils/plot_results.py
cp render_cartpole.py ../Utils/render_cartpole.py
mkdir ../Results_Cartpole/
mkdir ../Visualization/
cd ..
rm -rf ./modelBasedRL
