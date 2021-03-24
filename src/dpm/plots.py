import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
# plt.rcParams['animation.ffmpeg_path'] = u'/usr/bin/ffmpeg'
import dpm
import pickle


import pmne.rsa_old

plt.rcParams['animation.ffmpeg_path'] = u'/usr/local/bin/ffmpeg'

#------------------------------------------------------------------
def plot_evoked(evoked, cond_names, meg_channels=None):

    fig_ids = []

    vmin, vmax = -4, 4

    for i in range(len(cond_names)):
        print("Plotting figures for condition={:}".format(cond_names[i]))

        fig_top = evoked[i].plot_topomap(contours=0, vmin=vmin, vmax=vmax)
        fig_top.canvas.set_window_title('Evoked plot ({:})'.format(cond_names[i]))

        figs_e = evoked[i].plot_joint(ts_args=dict(spatial_colors=True))
        if isinstance(figs_e, plt.Figure):
            figs_e = figs_e,
        [f.canvas.set_window_title('Evoked plot ({:})'.format(cond_names[i])) for f in figs_e]

        cond_data = dict(condition=cond_names[i], topomap=fig_top)
        if meg_channels == 'mag':
            cond_data['grad'] = None
            cond_data['mag'] = figs_e[0]
        elif meg_channels == 'grad':
            cond_data['grad'] = figs_e[0]
            cond_data['mag'] = None
        elif meg_channels is None:
            cond_data['grad'] = figs_e[0]
            cond_data['mag'] = figs_e[1]
        else:
            raise Exception('Invalid "meg_channels" argument')

        fig_ids.append(cond_data)

    return fig_ids


#---------------------------------------------------------

def plot_dissimilarity(dissim,vmax = None):

    matrix = dissim.data

    min_val = 0
    if vmax is None:
        max_val = np.mean(matrix) + np.std(matrix)
    else:
        max_val = vmax

    plt.imshow(np.mean(matrix,axis=0), interpolation='none', cmap=cmap, origin='upper', vmin=min_val, vmax=max_val)
    plt.colorbar()
    x_ticks = extract_ticks_labels_from_md(dissim.md0)
    y_ticks = extract_ticks_labels_from_md(dissim.md1)
    y = range(matrix.shape[1])
    x = range(matrix.shape[2])
    plt.xticks(x,x_ticks,rotation='vertical')
    plt.yticks(y,y_ticks)



def extract_ticks_labels_from_md(metadata):

    xticks_labels = []
    for m in range(len(metadata)):
        string_fields = ''
        for field in metadata.keys():
            string_fields += '%s_%s_'%(field[:3],str(metadata[field][m]))

        xticks_labels.append(string_fields)

    return xticks_labels



def video_dissimilarity_matrices(dissimilarity, save_path_video, tmin=-500,tmax=700,interval = 100,vmin=None,vmax=None):


    data_dissimilarity = dissimilarity.data
    n_times,n_comps,n_comps = data_dissimilarity.shape


    time_stamps = np.round(np.linspace(tmin,tmax,n_times),3)

    if vmin is None:
        vmin = np.mean(data_dissimilarity) - np.std(data_dissimilarity)
    if vmax is None:
        max_val = np.mean(data_dissimilarity) + np.std(data_dissimilarity)
    else:
        max_val = vmax

    fig = plt.figure()
    fig = plt.figure(figsize=(20, 20))
    ax = plt.axes()
    ttl = ax.text(.5, 1.05, '', transform=ax.transAxes, va='center')

    def f(k, data_dissimilarity):
        return data_dissimilarity[k, :, :]


    im = plt.imshow((f(0, data_dissimilarity)), vmin=vmin, vmax=max_val, interpolation='none', animated=True,cmap=cmap)
    plt.colorbar()
    x_ticks = extract_ticks_labels_from_md(dissimilarity.md0)
    y_ticks = extract_ticks_labels_from_md(dissimilarity.md1)
    y = range(data_dissimilarity.shape[1])
    x = range(data_dissimilarity.shape[2])
    plt.xticks(x,x_ticks,rotation='vertical',fontsize=12)
    plt.yticks(y,y_ticks, fontsize=12)

    def init():
        ttl.set_text('')
        return ttl

    def updatefig(k):
        im.set_array(np.transpose(f(k %n_times , data_dissimilarity)))
        ttl.set_text('%i ms' % time_stamps[k])
        return im,

    ani = animation.FuncAnimation(fig, updatefig, init_func=init, frames=n_times-1, interval=interval, blit=False)
    plt.show()

    ani.save(save_path_video)

    return True


#---------------------------------------------------------

def plot_saved_decoding_score(dir_name, subjects, chance_level=0.5):

    all_scores = []

    for subj in subjects:
        filename = '{:}/scores_{:}.pkl'.format(dir_name, subj)
        with open(filename, 'r') as fp:
            subj_scores = pickle.load(fp)
        for s in subj_scores:
            s['subj'] = subj
            all_scores.append(s)

    times = all_scores[0]['times']
    scores = np.array([s['scores'] for s in all_scores]).mean(axis=0)
    score_stds = np.array([s['scores'] for s in all_scores]).std(axis=0)

    _do_plot_scores(times, scores, chance_level, sd=score_stds, n_subjects=len(subjects))

    return scores


#------------------------------------------------------------------
def _do_plot_scores(times, scores, chance_level, sd=None, n_subjects=None):

    ymin = np.floor(min(scores) * 10) / 10
    ymax = np.ceil(max(scores) * 10) / 10

    plt.figure()  # New figure

    plt.plot([times[0], times[-1]], [chance_level, chance_level], color='b')
    plt.hold(True)
    plt.plot([0, 0], [ymin, ymax], color='g')
    plt.hold(True)

    plt.plot(times, scores, color='black')

    if sd is not None:
        scores = np.array(scores)
        se = sd / np.sqrt(n_subjects)
        plt.gca().fill_between(x=times, y1=scores - se, y2=scores + se, alpha=0.2, color='black')

    plt.xlabel('Time (s)')
    plt.ylabel('Classif. score (%)')
    plt.ylim([ymin, ymax])

    plt.grid()




def video_dissimilarity(data_dissimilarity,vmin=-1,vmax=1):

    """
    video of the dissimilarity across time
    :param data_dissimilarity:
    :param vmin:
    :param vmax:
    :return:
    """
    n_times,n_comps,n_comps = data_dissimilarity.shape


    fig = plt.figure(figsize=(10, 10))
    ims = []
    for i in range(n_times):
        mat_to_plot =data_dissimilarity[i,:,:]
        im = plt.imshow(mat_to_plot, interpolation='nearest', vmin=vmin, vmax=vmax, animated=True)
        ims.append([im])


    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)


    return ani


# ------------ plotting the diagonal of the GAT ------------

def plot_decoding_in_time(subject_name,considered_analysis,decoding_method):


    if decoding_method == 'standard':

        path_fig = dpm.consts.results_dir+'/decoding/' + considered_analysis + '/plot_' + subject_name + '_' + decoding_method + '.png'
        path_data = dpm.consts.results_dir+'/decoding/'+ considered_analysis + '/scores_' + subject_name + '_' + decoding_method + '.pkl'

        with open(path_data, 'rb') as fid:
            results = pickle.load(fid)

        if considered_analysis == 'unit' or considered_analysis == 'decade':
            scores = np.mean([results[i]['scores'] for i in range(len(results))], axis=0)
        else:
            scores = results['scores']

        scores = np.mean(scores, axis=0)
        times = results[0]['times']

    else:
        tmin = np.round(np.linspace(-0.2, 0.6, 34), 3)
        tmax = np.round([t + 0.1 for t in tmin], 3)

        score_all_time_window = []
        times_analysis = []
        for tt in range(len(tmin)):
            tmi = tmin[tt]
            tma = tmax[tt]
            path_fig = dpm.consts.results_dir+ '/decoding/' + considered_analysis + '/plot_' + subject_name + '_' + decoding_method + '.png'
            path_data = dpm.consts.results_dir+ '/decoding/' + considered_analysis + '/scores_' + subject_name + '_' + decoding_method + "_%i.0_%i.0" % (
            tmi * 1000, tma * 1000) + '.pkl'
            with open(path_data, 'rb') as fid:
                results = pickle.load(fid)
            scores = results[0]['scores']
            scores = np.mean(scores)
            times = results[0]['times']

            score_all_time_window.append(scores)
            times_analysis.append((times['tmax'] + times['tmin']) / 2)

        times = np.round(np.asanyarray(times_analysis))
        scores = np.asarray(score_all_time_window)


    fig = plt.figure()
    plt.plot(times, scores)
    plt.plot(times, [0.5] * len(times), 'r--', lw=2)
    plt.xlabel('time')
    plt.ylabel('decoding accuracy')
    plt.title('%s for method %s ' % (considered_analysis, decoding_method))

    fig.savefig(path_fig)


    return fig

# ---------------- plot the dissimilarity matrix ----------------

def plot_dissimilarity_matrix(dissimilarity_func,function_description,targets,vmin=0,vmax=1):

    """
    Function to plot the predicted dissimilarity
    :param dissimilarity_func: the dissimilarity you want to plot
    :param targets: list of the stimuli events_ids
    :return: The figure
    """

    dissim_matrix = pmne.rsa_old.gen_predicted_dissimilarity(dissimilarity_func, targets).matrix
    fig = plt.figure()
    plt.imshow(dissim_matrix,interpolation='nearest',vmin= vmin,vmax=vmax)
    plt.title('Dissimilarity matrix for %s'%(function_description))
    plt.colorbar()
    plt.show()


    return fig