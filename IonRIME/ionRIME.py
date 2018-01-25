import numpy as np
from visibility_simulation import VisibilitySimulation
import sys, os
import yaml
from shutil import copyfile, rmtree
import git
import h5py


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

    if P['MC_sky'] is False:
        VS = VisibilitySimulation(P)
        VS.run()

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        out_name = "visibility.npz"

        np.savez(out_dir + out_name, Vis=VS.Vis, parameters=P, git_hash=sha)

    if P['MC_sky'] is True:
        N_skies = P['MC_sky_realizations']

        out_name = 'visibilities.h5'

        h5f = h5py.File(out_dir + out_name, 'w')

        for n in range(N_skies):
            VS = VisibilitySimulation(P)
            VS.run()
            h5f.create_dataset('ionpol' + str(n), data=VS.Vis)

            VS.ndays = 1
            VS.ionosphere = 'none'
            VS.final_day_average = False

            VS.Vis = np.zeros(1 * VS.ntime * VS.nfreq * 2 * 2, dtype='complex128')
            VS.Vis = VS.Vis.reshape(1, VS.ntime, VS.nfreq, 2, 2)

            VS.run()
            h5f.create_dataset('pol' + str(n), data=VS.Vis)

            VS.Vis = np.zeros(1 * VS.ntime * VS.nfreq * 2 * 2, dtype='complex128')
            VS.Vis = VS.Vis.reshape(1, VS.ntime, VS.nfreq, 2, 2)

            VS.unpolarized = True
            VS.Q *= 0
            VS.U *= 0
            VS.V *= 0

            VS.Q_alm *= 8
            VS.U_alm *= 0

            VS.run()
            h5f.create_dataset('unpol' + str(n), data=VS.Vis)

        h5f.close()

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        np.savez(out_dir + 'meta_data.npz', parameters=P, git_hash=sha)
