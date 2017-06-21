import numpy as np
from visibility_simulation import VisibilitySimulation
import sys, os
import yaml
from shutil import copyfile, rmtree
import git


if __name__ == "__main__":

    parameters_file_name = sys.argv[1]

    yamlfile = file(parameters_file_name, 'r')
    P = yaml.load(yamlfile)

    for key in P:
        print key,":",P[key],":",type(P[key])

    if (P['unpolarized'] == True) and (P['ionosphere'] != 'none'):
        raise Exception('Simulation includes the ionosphere with an unpolarized sky! Dummy!')

    if P['unpolarized'] == False and P['ionosphere'] == 'none' and P['ndays'] > 1:
        raise Exception('Multiple days are set for a non-ionosphere run.')

    if P['instrument'] == 'paper':
        out_dir = os.path.join(P['output_base_directory'], 'PAPER/', P['output_directory'])
    elif P['instrument'] == 'paper_hfss':
        out_dir = os.path.join(P['output_base_directory'], 'PAPER/', P['output_directory'])
    elif P['instrument'] == 'paper_feko':
        out_dir = os.path.join(P['output_base_directory'], 'PAPER/', P['output_directory'])
    else:
        out_dir = os.path.join(P['output_base_directory'], 'HERA/', P['output_directory'])

    if P['debuging_mode'] is True:
        if os.path.exists(out_dir):
            rmtree(out_dir)

    while os.path.exists(out_dir):
        out_dir = out_dir[:-1] + "B/"

    os.makedirs(out_dir)
    copyfile(parameters_file_name, out_dir + 'parameters.yaml')
    print "Parameters file copied."

    if P['unpolarized'] is True:
        print "Polarization turned off"

    if P['ionosphere'] == 'none':
        print "Ionospheric rotation turned off"

    if P['circular_pol'] is False:
        print "No circular polarization! Booooo."

    VS = VisibilitySimulation(P)
    VS.run()

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    out_name = "visibility.npz"

    np.savez(out_dir + out_name, Vis=VS.Vis, parameters=P, git_hash=sha)
