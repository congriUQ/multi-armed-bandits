"""
This module simulates some Multi-Armed Bandit algorithms and implements an interactive game where the user can try to
beat the algorithms
"""
import pandas as pd
import numpy as np
from numpy.random import default_rng
# Pass a number to default_rng() if you want to fix the seed
rng = default_rng()
import scipy.special as spsp
import scipy.stats as spst
# from tqdm import tqdm
from IPython.display import display, HTML
import cProfile
import pstats
import io
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category10_10


class Agent:
    """
    An Agent can track the data, update the configuration, and show a teaser to the reader
    """

    def __init__(self, n_views_update_data=10000, n_views_update_configuration=1000, n_teasers=4, verbose=False):
        self.n_views_update_data = n_views_update_data
        self.n_views_update_configuration = n_views_update_configuration
        # This is the data the Agent is aware of, i.e., clicks and views after daily import
        self.n_teasers = n_teasers
        self.n_clicks = np.zeros(self.n_teasers, dtype=int)
        self.n_views = np.zeros(self.n_teasers, dtype=int)
        # these are hidden to the agent until the data is updated (= daily import)
        self.n_clicks_run = np.zeros(self.n_teasers, dtype=int)
        self.n_views_run = np.zeros(self.n_teasers, dtype=int)
        # pandas DataFrame for pretty output
        self.teaser_data = pd.DataFrame({'active': self.n_teasers * [True], 'n_clicks': 0, 'n_views': 0,
                                         'empirical ctr': None},
                                        index=[f'teaser {i}' for i in range(1, self.n_teasers + 1)])
        self.weights = np.ones(self.n_teasers)/self.n_teasers
        self.name = None
        self.verbose = verbose

    def update_data(self):
        self.n_clicks += self.n_clicks_run
        self.n_views += self.n_views_run
        self.n_views_run = np.zeros(self.n_teasers, dtype=int)
        self.n_clicks_run = np.zeros(self.n_teasers, dtype=int)
        if self.verbose:
            print('New data was imported!')

    def reset(self):
        self.n_views = np.zeros(self.n_teasers, dtype=int)
        self.n_clicks = np.zeros(self.n_teasers, dtype=int)
        self.n_views_run = np.zeros(self.n_teasers, dtype=int)
        self.n_clicks_run = np.zeros(self.n_teasers, dtype=int)
        self.teaser_data['empirical ctr'] = None
        self.teaser_data['active'] = True

    def update_teaser_data_df(self):
        self.teaser_data['active'] = self.weights.astype(bool)
        self.teaser_data['n_clicks'] = self.n_clicks
        self.teaser_data['n_views'] = self.n_views
        for i in range(self.n_teasers):
            if self.n_views[i] > 0:
                self.teaser_data.loc[f'teaser {i + 1}', 'empirical ctr'] = \
                    '%2.2f' % (100.0 * self.n_clicks[i]/self.n_views[i]) + '%'
            else:
                self.teaser_data.loc[f'teaser {i + 1}', 'empirical ctr'] = None

        # if self.n_views.all():
        #     self.teaser_data['empirical ctr'] = 100.0 * self.n_clicks/self.n_views
        #     self.teaser_data['empirical ctr'] = self.teaser_data['empirical ctr'].apply(lambda x: '%2.2f' % x + '%')
        # else:
        #     self.teaser_data['empirical ctr'] = None

    def update_configuration(self):
        pass

    def show_teaser(self):
        teaser_id = rng.choice(self.n_teasers, p=self.weights)
        return teaser_id


class HumanAgent(Agent):

    def __init__(self, n_teasers=4, n_views_update_data=10000, n_views_update_configuration=70000):
        super().__init__(n_teasers=n_teasers, n_views_update_data=n_views_update_data,
                         n_views_update_configuration=n_views_update_configuration)
        self.name = f'human_agent_update_data={n_views_update_data}_update_config={n_views_update_configuration}'

    def update_configuration(self):
        print('The teaser clicks and views you know from daily import are:')
        self.update_teaser_data_df()
        display(self.teaser_data)
        valid_config = False
        while not valid_config:
            self.weights = np.zeros(self.n_teasers)
            for t in range(self.n_teasers):
                print(f'Do you want to turn on {self.teaser_data.index[t]} (y/n)?')
                active = input()
                if active.lower() in ['yes', '1', 'y', 'true']:
                    self.weights[t] = 1.0
            self.weights = self.weights/self.weights.sum()
            if np.absolute(self.weights.sum() - 1.0) < 1e-9:
                valid_config = True
                print('Configuration updated, continue running...\n')
            else:
                print('WARNING: At least one teaser needs to be active. Please set the configuration again!')


class NoOptimizationAgent(Agent):
    """
    This agent does nothing
    """

    def __init__(self, n_teasers=4):
        super().__init__(n_teasers=n_teasers)
        self.name = 'no_optimization_agent'


class ThompsonSamplingAgent(Agent):

    def __init__(self, prior_alpha=4.0, prior_beta=800.0, n_teasers=4, n_views_update_data=10000,
                 n_views_update_configuration=1000, exploitation_factor=1.0, verbose=False):
        super().__init__(n_teasers=n_teasers, n_views_update_data=n_views_update_data,
                         n_views_update_configuration=n_views_update_configuration, verbose=verbose)
        self.name = f'thompson_agent_update_data={n_views_update_data}_update_config={n_views_update_configuration}'
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.alpha = self.prior_alpha
        self.beta = self.prior_beta
        self.chosen_teaser = None
        self.update_before_data = False  # modify configuration before any tracking data is available?
        self.exploitation_factor = exploitation_factor

    def update_posteriors(self):
        self.alpha = self.prior_alpha + self.exploitation_factor * self.n_clicks
        self.beta = self.prior_beta + self.exploitation_factor * (self.n_views - self.n_clicks)

    def reset(self):
        self.n_views = np.zeros(self.n_teasers, dtype=int)
        self.n_clicks = np.zeros(self.n_teasers, dtype=int)
        self.n_views_run = np.zeros(self.n_teasers, dtype=int)
        self.n_clicks_run = np.zeros(self.n_teasers, dtype=int)
        self.alpha = self.prior_alpha
        self.beta = self.prior_beta
        self.chosen_teaser = None

    def update_configuration(self):
        # Thompson Sampling: Sample ctr's from posteriors and pick the best teaser
        self.update_posteriors()  # make sure we work with the latest data

        if self.update_before_data or self.n_views.sum() > 0:
            ctr_samples = rng.beta(a=self.alpha, b=self.beta)
            self.chosen_teaser = np.argmax(ctr_samples)
            self.weights = np.zeros(self.n_teasers)
            self.weights[self.chosen_teaser] = 1.0
            if self.verbose:
                self.plot_action(ctr_samples)

    def show_teaser(self):
        if self.chosen_teaser is not None:
            return self.chosen_teaser
        else:
            # do this before any data was observed
            return rng.choice(self.n_teasers, p=self.weights)

    def plot_action(self, ctr_samples):
        self.update_teaser_data_df()
        print('Known data after daily import:')
        display(self.teaser_data)

        ctr = np.linspace(.0, .015, 501)
        p_ctr = []
        for i in range(self.n_teasers):
            p_ctr.append(spst.beta.pdf(ctr, self.alpha[i], self.beta[i]))
        fig = figure(background_fill_color=(255, 255, 255), plot_width=1000, plot_height=300,
                   tools="hover,pan,wheel_zoom,box_zoom,reset,crosshair,save", toolbar_location='right',
                   title="Thompson Sampling Configuration Update")

        lines = []
        circles = []
        for i, dens in enumerate(p_ctr):
            line_width = 5 if i == self.chosen_teaser else 2
            line = fig.line(ctr, dens, legend_label=self.teaser_data.index[i], color=Category10_10[i],
                            line_width=line_width)
            circle = fig.circle(x=ctr_samples[i], y=16.0, fill_color=Category10_10[i],
                                line_width=0, size=15)
            lines.append(line)
            circles.append(circle)
        chosen_circle = fig.circle(x=ctr_samples[self.chosen_teaser], y=16.0, fill_alpha=0.0, size=35,
                                   line_color='black', line_width=5, legend_label='chosen teaser')

        fig.y_range.start = 0.0
        fig.xaxis.major_label_text_font_size = "16px"
        fig.yaxis.major_label_text_font_size = "16px"
        fig.yaxis.axis_label = 'p(ctr)'
        fig.xaxis.axis_label = 'ctr'
        fig.legend.location = 'bottom_right'
        show(fig)
        print('Chosen teaser = ', self.teaser_data.index.to_list()[self.chosen_teaser])
        input('Hit enter to continue\n')


class UCBAgent(Agent):

    def __init__(self, prior_alpha=4.0, prior_beta=800.0, n_teasers=4, n_views_update_data=10000,
                 n_views_update_configuration=1000, ucb_quantile=.95, verbose=False):
        super().__init__(n_teasers=n_teasers, n_views_update_data=n_views_update_data,
                         n_views_update_configuration=n_views_update_configuration, verbose=verbose)
        self.name = f'ucb_agent_update_data={n_views_update_data}_update_config={n_views_update_configuration}'
        self.prior_alpha = prior_alpha * np.ones(self.n_teasers)
        self.prior_beta = prior_beta * np.ones(self.n_teasers)
        self.ucb_quantile = ucb_quantile
        self.alpha = self.prior_alpha
        self.beta = self.prior_beta
        self.chosen_teaser = None
        self.update_before_data = False  # modify configuration before any tracking data is available?

    def update_posteriors(self):
        self.alpha = self.prior_alpha + self.n_clicks
        self.beta = self.prior_beta + self.n_views - self.n_clicks

    def reset(self):
        self.n_views = np.zeros(self.n_teasers, dtype=int)
        self.n_clicks = np.zeros(self.n_teasers, dtype=int)
        self.n_views_run = np.zeros(self.n_teasers, dtype=int)
        self.n_clicks_run = np.zeros(self.n_teasers, dtype=int)
        self.alpha = self.prior_alpha
        self.beta = self.prior_beta
        self.chosen_teaser = None

    def update_configuration(self):
        # Thompson Sampling: Sample ctr's from posteriors and pick the best teaser
        self.update_posteriors()  # make sure we work with the latest data

        if self.update_before_data or self.n_views.sum() > 0:
            upper_confidence_bounds = spsp.betaincinv(self.alpha, self.beta, self.ucb_quantile)
            self.chosen_teaser = np.argmax(upper_confidence_bounds)
            self.weights = np.zeros(self.n_teasers)
            self.weights[self.chosen_teaser] = 1.0
            if self.verbose:
                self.plot_action(upper_confidence_bounds)

    def show_teaser(self):
        if self.chosen_teaser is not None:
            return self.chosen_teaser
        else:
            # do this before any data was observed
            return rng.choice(self.n_teasers, p=self.weights)

    def plot_action(self, upper_confidence_bounds):
        self.update_teaser_data_df()
        print('Known data after daily import:')
        display(self.teaser_data)

        ctr = np.linspace(.0, .015, 501)
        p_ctr = []
        for i in range(self.n_teasers):
            p_ctr.append(spst.beta.pdf(ctr, self.alpha[i], self.beta[i]))
        fig = figure(background_fill_color=(255, 255, 255), plot_width=1000, plot_height=300,
                   tools="hover,pan,wheel_zoom,box_zoom,reset,crosshair,save", toolbar_location='right',
                   title="Thompson Sampling Configuration Update")

        lines = []
        circles = []
        for i, dens in enumerate(p_ctr):
            line_width = 5 if i == self.chosen_teaser else 2
            line = fig.line(ctr, dens, legend_label=self.teaser_data.index[i], color=Category10_10[i],
                            line_width=line_width)
            circle = fig.circle(x=upper_confidence_bounds[i], y=16.0, fill_color=Category10_10[i],
                                line_width=0, size=15)
            lines.append(line)
            circles.append(circle)
        chosen_circle = fig.circle(x=upper_confidence_bounds[self.chosen_teaser], y=16.0, fill_alpha=0.0, size=35,
                                   line_color='black', line_width=5, legend_label='chosen teaser')

        fig.y_range.start = 0.0
        fig.xaxis.major_label_text_font_size = "16px"
        fig.yaxis.major_label_text_font_size = "16px"
        fig.yaxis.axis_label = 'p(ctr)'
        fig.xaxis.axis_label = 'ctr'
        fig.legend.location = 'bottom_right'
        show(fig)
        print('Chosen teaser = ', self.teaser_data.index.to_list()[self.chosen_teaser])
        input('Hit enter to continue\n')


class EpsilonGreedyAgent(Agent):

    def __init__(self, epsilon=.05, n_teasers=4, n_views_update_data=10000, n_views_update_configuration=1000,
                 verbose=False):
        super().__init__(n_teasers=n_teasers, n_views_update_data=n_views_update_data,
                         n_views_update_configuration=n_views_update_configuration, verbose=verbose)
        self.name = f'{epsilon}_greedy_agent_update_data={n_views_update_data}_update_config' \
                    f'={n_views_update_configuration}'
        self.epsilon = epsilon
        self.update_before_data = False  # modify configuration before any tracking data is available?
        self.chosen_teaser = None

    def update_configuration(self):
        if self.update_before_data or self.n_views.sum() > 0:
            # With 1 - epsilon probability, use the best teaser. With epsilon probability, use one of the other teasers
            ctr = self.n_clicks/self.n_views
            best_teaser = np.argmax(ctr)
            p_teaser = (self.epsilon/(self.n_teasers - 1)) * np.ones(self.n_teasers)
            p_teaser[best_teaser] = 1.0 - self.epsilon
            self.chosen_teaser = rng.choice(self.n_teasers, p=p_teaser)
            self.weights = np.zeros(self.n_teasers)
            self.weights[best_teaser] = 1.0
            if self.verbose:
                self.plot_action()

    def show_teaser(self):
        if self.chosen_teaser is not None:
            return self.chosen_teaser
        else:
            # do this before any data was observed
            return rng.choice(self.n_teasers, p=self.weights)

    def plot_action(self):
        self.update_teaser_data_df()
        print('Known data after daily import:')
        display(self.teaser_data)

        emp_ctr = 100.0 * self.n_clicks/self.n_views
        colors = self.n_teasers * [Category10_10[0]]
        colors[self.chosen_teaser] = Category10_10[1]
        legend_group = self.n_teasers * ['not chosen']
        legend_group[self.chosen_teaser] = 'chosen'
        source = ColumnDataSource({'ctr': emp_ctr, 'colors': colors, 'label': legend_group,
                                   'teaser': self.teaser_data.index.to_list()})
        fig = figure(background_fill_color=(255, 255, 255), plot_width=500, plot_height=250,
                   tools="hover,pan,wheel_zoom,box_zoom,reset,crosshair,save", toolbar_location='right',
                   title="Epsilon Greedy Configuration Update", x_range=self.teaser_data.index.to_list())
        bars = fig.vbar(x='teaser', top='ctr', width=0.8, color='colors',
                        legend_group='label', source=source)
        fig.y_range.start = 0.0
        fig.xaxis.major_label_text_font_size = "16px"
        fig.xaxis.major_label_orientation = .4
        fig.yaxis.major_label_text_font_size = "16px"
        fig.yaxis.axis_label = 'Teaser CTR in %'
        fig.legend.location = 'bottom_right'
        show(fig)
        print('Chosen teaser = ', self.teaser_data.index.to_list()[self.chosen_teaser])
        input('Hit enter to continue\n')


class OptimizationGame:

    def __init__(self, agents, true_ctrs, n_views_total=int(1e5)):
        self.agents = agents
        self.n_views_total = n_views_total
        self.true_ctrs = true_ctrs

    def print_cum_stats(self):
        print('The cumulative stats are:\n')
        for agent in self.agents:
            c = agent.n_clicks.sum() + agent.n_clicks_run.sum()
            v = agent.n_views.sum() + agent.n_views_run.sum()
            print(agent.name + ':')
            print(f'clicks:        {c}\n'
                  f'views:         {v}\n'
                  f'empirical ctr: {100 * c / v}\n')

    def _run(self):
        for n_views_tot in range(self.n_views_total):
            for agent in self.agents:
                if (n_views_tot % agent.n_views_update_configuration) == 0:
                    if agent.__class__.__name__ == 'HumanAgent' and (n_views_tot > 0):
                        self.print_cum_stats()
                    agent.update_configuration()
                if (n_views_tot % agent.n_views_update_data) == 0:
                    agent.update_data()

                shown_teaser = agent.show_teaser()
                click = int(rng.binomial(1, self.true_ctrs[shown_teaser]))
                agent.n_clicks_run[shown_teaser] += click
                agent.n_views_run[shown_teaser] += 1

        print('Game over.')
        print('Stats:')
        for agent in self.agents:
            print('\n\n' + agent.name + ':')
            agent.update_teaser_data_df()
            display(agent.teaser_data)

        print('\n\n\n\n')
        self.print_cum_stats()

    def start(self):
        for agent in self.agents:
            agent.reset()
        self._run()

    def resume(self):
        self._run()
