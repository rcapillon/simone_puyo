"""
streamlit_app.py
================
Interface Streamlit pour analyser une partie de Puyo sauvegardée par play_and_save.py.

Lancement :
    streamlit run streamlit_app.py -- --replay ./game_replay.pkl

Ou via l'interface : uploader le fichier .pkl directement dans l'app.
"""

import io
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import streamlit as st
import pandas as pd

# ── Optionally load replay path from CLI args ──────────────────────────────
DEFAULT_REPLAY_PATH = None
if '--replay' in sys.argv:
    idx = sys.argv.index('--replay')
    if idx + 1 < len(sys.argv):
        DEFAULT_REPLAY_PATH = sys.argv[idx + 1]

# ── Puyo color map (mirrors puyo.py) ──────────────────────────────────────
cvals  = [0, 1, 2, 3, 4, 5]
colors = ["c", "r", "g", "b", "m", "white"]
norm   = plt.Normalize(min(cvals), max(cvals))
color_tuples = list(zip(map(norm, cvals), colors))
puyo_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", color_tuples)

ACTION_LABELS = (
    [f"V↑ col{c}" for c in range(6)] +   # 0-5  : vertical, puyo1 bottom
    [f"V↓ col{c}" for c in range(6)] +   # 6-11 : vertical, puyo2 bottom
    [f"H→ {c}-{c+1}" for c in range(5)] +# 12-16: horizontal, puyo1 left
    [f"H← {c}-{c+1}" for c in range(5)]  # 17-21: horizontal, puyo2 left
)


# ── Helpers ────────────────────────────────────────────────────────────────

def render_board_fig(board_num, queue_num, title=""):
    """Return a matplotlib Figure showing board + queue (same as display_screen)."""
    fig, ax = plt.subplots(figsize=(3.5, 5))
    screen = np.zeros((13, 8))
    screen[:, :6] = board_num
    screen[:, 6]  = 5.
    screen[0, 7]  = queue_num[0, 0]
    screen[1, 7]  = queue_num[0, 1]
    screen[2, 7]  = 5.
    screen[3, 7]  = queue_num[1, 0]
    screen[4, 7]  = queue_num[1, 1]
    screen[5, 7]  = 5.
    screen[6, 7]  = queue_num[2, 0]
    screen[7, 7]  = queue_num[2, 1]
    screen[8:, 7] = 5.
    ax.imshow(screen, cmap=puyo_cmap, vmin=0, vmax=5)
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    fig.tight_layout(pad=0.3)
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return buf


def render_policy_bar(policy, legal_actions, chosen_action, title="Policy"):
    """Bar chart of MCTS policy over legal actions, highlighting the chosen one."""
    fig, ax = plt.subplots(figsize=(7, 2.5))
    x      = np.arange(22)
    colors_bar = ['#e74c3c' if i == chosen_action
                  else '#3498db' if i in legal_actions
                  else '#bdc3c7'
                  for i in x]
    ax.bar(x, policy, color=colors_bar, edgecolor='none')
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_LABELS, rotation=90, fontsize=6)
    ax.set_ylabel('Visit probability', fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, max(policy.max() * 1.15, 0.05))
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(pad=0.3)
    return fig


def render_value_history(steps, current_step):
    """Line chart of network value and MCTS value across all steps."""
    fig, ax = plt.subplots(figsize=(7, 2.5))
    xs = list(range(len(steps)))
    mcts_vals = [s.mcts_value    for s in steps]
    net_vals  = [s.network_value for s in steps]
    rewards   = [s.reward        for s in steps]

    ax.plot(xs, mcts_vals,  label='MCTS value',    color='#2ecc71', linewidth=1.5)
    ax.plot(xs, net_vals,   label='Network value', color='#3498db', linewidth=1.5, linestyle='--')
    ax.axvline(current_step, color='#e74c3c', linewidth=1.5, linestyle=':', label='Current step')
    # Scatter rewards as stem points
    for i, r in enumerate(rewards):
        if r != 0:
            ax.scatter(i, r, marker='D', color='#f39c12', zorder=5, s=30)

    ax.set_xlabel('Step', fontsize=8)
    ax.set_ylabel('Value / Reward', fontsize=8)
    ax.set_title('Value estimates & rewards over time  (◆ = non-zero reward)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout(pad=0.3)
    return fig


def render_returns_history(replay, current_step):
    fig, ax = plt.subplots(figsize=(7, 2.5))
    xs = list(range(len(replay.discounted_returns)))
    ax.plot(xs, replay.discounted_returns, color='#9b59b6', linewidth=1.5, label='Discounted return')
    ax.axvline(current_step, color='#e74c3c', linewidth=1.5, linestyle=':', label='Current step')
    ax.set_xlabel('Step', fontsize=8)
    ax.set_ylabel('G_t', fontsize=8)
    ax.set_title(f'Discounted returns (γ={replay.discount_factor})', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout(pad=0.3)
    return fig


def render_chain_histogram(steps):
    chains = [s.chain_length for s in steps]
    fig, ax = plt.subplots(figsize=(4, 2.5))
    bins = range(0, max(chains) + 2)
    ax.hist(chains, bins=bins, align='left', color='#1abc9c', edgecolor='white', rwidth=0.8)
    ax.set_xlabel('Chain length', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.set_title('Distribution of chain lengths', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(pad=0.3)
    return fig


def render_entropy_history(steps, current_step):
    """Policy entropy over time — high entropy = uncertain agent."""
    def entropy(p):
        p = np.clip(p, 1e-9, 1)
        return -np.sum(p * np.log(p))

    mcts_ent = [entropy(s.mcts_policy)    for s in steps]
    net_ent  = [entropy(s.network_policy) for s in steps]
    xs = list(range(len(steps)))

    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.plot(xs, mcts_ent,  label='MCTS policy entropy',    color='#2ecc71', linewidth=1.5)
    ax.plot(xs, net_ent,   label='Network policy entropy', color='#3498db', linewidth=1.5, linestyle='--')
    ax.axvline(current_step, color='#e74c3c', linewidth=1.5, linestyle=':')
    ax.set_xlabel('Step', fontsize=8)
    ax.set_ylabel('Entropy (nats)', fontsize=8)
    ax.set_title('Policy entropy over time', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout(pad=0.3)
    return fig


def build_step_table(steps):
    rows = []
    for s in steps:
        rows.append({
            'Step':          s.step,
            'Action':        ACTION_LABELS[s.action],
            'Chain':         s.chain_length,
            'Reward':        round(s.reward, 2),
            'MCTS value':    round(s.mcts_value, 3),
            'Net value':     round(s.network_value, 3),
            'Policy @ action': round(float(s.mcts_policy[s.action]), 4),
            'Game over':     s.gameover,
        })
    return pd.DataFrame(rows)


# ── App ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title='Puyo Replay Analyser', layout='wide', page_icon='🟢')

st.title('🟢 Puyo Puyo — Replay Analyser')

# ── Load replay ────────────────────────────────────────────────────────────
replay = None

with st.sidebar:
    st.header('📂 Load replay')
    uploaded = st.file_uploader('Upload a .pkl replay file', type=['pkl'])
    if uploaded is not None:
        replay = pickle.load(uploaded)
        st.success('Replay loaded from upload.')
    elif DEFAULT_REPLAY_PATH:
        try:
            with open(DEFAULT_REPLAY_PATH, 'rb') as f:
                replay = pickle.load(f)
            st.success(f'Loaded: {DEFAULT_REPLAY_PATH}')
        except FileNotFoundError:
            st.error(f'File not found: {DEFAULT_REPLAY_PATH}')

if replay is None:
    st.info('👈 Upload a replay file (generated by `play_and_save.py`) to get started.')
    st.stop()

steps = replay.steps
n_steps = len(steps)

# ── Sidebar: game summary ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('---')
    st.subheader('📊 Game summary')
    st.metric('Agent',         f'{replay.agent_type} — {replay.agent_name}')
    st.metric('Total steps',   replay.n_steps)
    st.metric('Total reward',  f'{replay.total_reward:.2f}')
    st.metric('Game over',     '💀 Yes' if replay.gameover else '✅ No (max moves)')
    chains = [s.chain_length for s in steps]
    st.metric('Best chain',    max(chains))
    st.metric('Chain moves',   f'{sum(c > 0 for c in chains)} / {n_steps}')
    st.metric('Mean value (MCTS)', f'{np.mean([s.mcts_value for s in steps]):.3f}')

    st.markdown('---')
    st.subheader('⚙️ MCTS config')
    cfg = replay.mcts_config
    st.write(f'Simulations : **{cfg.n_simulations}**')
    st.write(f'UCT c       : **{cfg.UCT_exploration_constant}**')
    st.write(f'γ (discount): **{cfg.discount_factor}**')
    st.write(f'Dir. α / ε  : **{cfg.dirichlet_alpha}** / **{cfg.dirichlet_epsilon}**')

# ── Tabs ───────────────────────────────────────────────────────────────────
tab_step, tab_overview, tab_table = st.tabs(['🔍 Step-by-step', '📈 Overview', '📋 Full table'])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — Step-by-step viewer
# ─────────────────────────────────────────────────────────────────────────
with tab_step:
    st.subheader('Step-by-step analysis')

    step_idx = st.slider(
        'Move', min_value=0, max_value=n_steps - 1, value=0, step=1,
        format='Step %d'
    )
    rec = steps[step_idx]

    # ── Row 1 : Board  |  Policy  |  KPIs ──────────────────────────────
    col_board, col_policy, col_kpi = st.columns([1.5, 3.5, 2])

    with col_board:
        chain_label = f'  ⛓ chain {rec.chain_length}' if rec.chain_length else ''
        fig_board = render_board_fig(
            rec.board_num, rec.queue_num,
            title=f'Step {step_idx}{chain_label}'
        )
        st.pyplot(fig_board, use_container_width=False)
        plt.close(fig_board)

    with col_policy:
        fig_pol = render_policy_bar(
            rec.mcts_policy, rec.legal_actions, rec.action,
            title=f'MCTS policy  (chosen: {ACTION_LABELS[rec.action]})'
        )
        st.pyplot(fig_pol, use_container_width=True)
        plt.close(fig_pol)

        # Raw network policy below
        fig_net = render_policy_bar(
            rec.network_policy, rec.legal_actions, rec.action,
            title='Network policy (raw, before MCTS)'
        )
        st.pyplot(fig_net, use_container_width=True)
        plt.close(fig_net)

    with col_kpi:
        st.markdown('#### Move details')
        st.metric('Chosen action',  ACTION_LABELS[rec.action])
        st.metric('Reward',         f'{rec.reward:.2f}')
        st.metric('Chain length',   rec.chain_length)
        st.metric('MCTS value',     f'{rec.mcts_value:.4f}')
        st.metric('Network value',  f'{rec.network_value:.4f}')
        disc_ret = replay.discounted_returns[step_idx] if replay.discounted_returns else 'N/A'
        st.metric('Discounted G_t', f'{disc_ret:.4f}' if isinstance(disc_ret, float) else disc_ret)
        st.metric('Value error',    f'{abs(rec.mcts_value - disc_ret):.4f}' if isinstance(disc_ret, float) else 'N/A')

        # Tsumo queue
        q = rec.queue_num
        color_names = {0: '—', 1: 'Cyan', 2: 'Red', 3: 'Green', 4: 'Blue'}
        st.markdown('#### Queue')
        st.write(f'Current : {color_names[q[0,0]]} / {color_names[q[0,1]]}')
        st.write(f'Next 1  : {color_names[q[1,0]]} / {color_names[q[1,1]]}')
        st.write(f'Next 2  : {color_names[q[2,0]]} / {color_names[q[2,1]]}')

        # Legal actions list
        with st.expander('Legal actions'):
            st.write([ACTION_LABELS[a] for a in rec.legal_actions])

    # ── Row 2 : Value + Returns history ────────────────────────────────
    st.markdown('---')
    col_val, col_ret = st.columns(2)
    with col_val:
        fig_val = render_value_history(steps, step_idx)
        st.pyplot(fig_val, use_container_width=True)
        plt.close(fig_val)
    with col_ret:
        fig_ret = render_returns_history(replay, step_idx)
        st.pyplot(fig_ret, use_container_width=True)
        plt.close(fig_ret)

    # ── Row 3 : Entropy history ─────────────────────────────────────────
    fig_ent = render_entropy_history(steps, step_idx)
    st.pyplot(fig_ent, use_container_width=True)
    plt.close(fig_ent)


# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — Global overview
# ─────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.subheader('Full-game overview')

    col_l, col_r = st.columns(2)

    with col_l:
        # Reward per step
        fig_rwd, ax_rwd = plt.subplots(figsize=(6, 2.5))
        rewards = [s.reward for s in steps]
        ax_rwd.bar(range(n_steps), rewards,
                   color=['#e74c3c' if r < 0 else '#2ecc71' for r in rewards])
        ax_rwd.set_xlabel('Step', fontsize=8)
        ax_rwd.set_ylabel('Reward', fontsize=8)
        ax_rwd.set_title('Reward per step', fontsize=9)
        ax_rwd.grid(axis='y', alpha=0.3)
        fig_rwd.tight_layout(pad=0.3)
        st.pyplot(fig_rwd, use_container_width=True)
        plt.close(fig_rwd)

        # Chain length over time
        fig_ch, ax_ch = plt.subplots(figsize=(6, 2.5))
        ax_ch.step(range(n_steps), chains, where='mid', color='#9b59b6', linewidth=1.5)
        ax_ch.fill_between(range(n_steps), chains, step='mid', alpha=0.2, color='#9b59b6')
        ax_ch.set_xlabel('Step', fontsize=8)
        ax_ch.set_ylabel('Chain', fontsize=8)
        ax_ch.set_title('Chain length per step', fontsize=9)
        ax_ch.grid(alpha=0.3)
        fig_ch.tight_layout(pad=0.3)
        st.pyplot(fig_ch, use_container_width=True)
        plt.close(fig_ch)

    with col_r:
        # Chain distribution
        fig_hist = render_chain_histogram(steps)
        st.pyplot(fig_hist, use_container_width=True)
        plt.close(fig_hist)

        # Action distribution
        fig_act, ax_act = plt.subplots(figsize=(6, 2.5))
        action_counts = np.zeros(22)
        for s in steps:
            action_counts[s.action] += 1
        ax_act.bar(range(22), action_counts, color='#3498db')
        ax_act.set_xticks(range(22))
        ax_act.set_xticklabels(ACTION_LABELS, rotation=90, fontsize=5.5)
        ax_act.set_ylabel('Count', fontsize=8)
        ax_act.set_title('Action usage distribution', fontsize=9)
        ax_act.grid(axis='y', alpha=0.3)
        fig_act.tight_layout(pad=0.3)
        st.pyplot(fig_act, use_container_width=True)
        plt.close(fig_act)

    # Value + entropy full-game
    st.markdown('---')
    fig_val_all = render_value_history(steps, -1)
    st.pyplot(fig_val_all, use_container_width=True)
    plt.close(fig_val_all)

    fig_ent_all = render_entropy_history(steps, -1)
    st.pyplot(fig_ent_all, use_container_width=True)
    plt.close(fig_ent_all)

    # Value error (|MCTS value - discounted return|)
    if replay.discounted_returns:
        fig_err, ax_err = plt.subplots(figsize=(7, 2.5))
        errors = [abs(s.mcts_value - g) for s, g in zip(steps, replay.discounted_returns)]
        ax_err.plot(errors, color='#e67e22', linewidth=1.5)
        ax_err.set_xlabel('Step', fontsize=8)
        ax_err.set_ylabel('|MCTS value - G_t|', fontsize=8)
        ax_err.set_title('Value estimation error over time', fontsize=9)
        ax_err.grid(alpha=0.3)
        fig_err.tight_layout(pad=0.3)
        st.pyplot(fig_err, use_container_width=True)
        plt.close(fig_err)


# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — Full data table
# ─────────────────────────────────────────────────────────────────────────
with tab_table:
    st.subheader('Per-step data table')
    df = build_step_table(steps)
    st.dataframe(df, use_container_width=True, height=500)

    # Download as CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='⬇️ Download as CSV',
        data=csv,
        file_name='puyo_replay.csv',
        mime='text/csv'
    )