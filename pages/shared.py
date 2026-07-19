"""
shared.py — cross-tab autofill for shared patient fields.

Any field entered on one tab auto-fills the same field on every other tab.

How it works: each shared widget syncs to a canonical profile dict via an
on_change callback that also bumps a version counter. Before a widget renders,
if the profile is newer than the value that widget last saw, its state is
re-seeded from the profile. Callbacks fire before Streamlit's rerun, so edits
propagate without clobbering the field the user just typed in.
"""
import streamlit as st


def profile():
    return st.session_state.setdefault("profile", {})


def _seed(key, field, default):
    pv = st.session_state.get("profile_version", 0)
    seen = f"_seen::{key}"
    prof = profile()
    if key not in st.session_state:
        st.session_state[key] = prof.get(field, default)
        st.session_state[seen] = pv
    elif st.session_state.get(seen, -1) < pv and field in prof:
        st.session_state[key] = prof[field]
        st.session_state[seen] = pv


def _cb(key, field):
    profile()[field] = st.session_state[key]
    st.session_state["profile_version"] = st.session_state.get("profile_version", 0) + 1


def number(label, field, key, default, min_value, max_value, step=1.0, **kw):
    _seed(key, field, default)
    return st.number_input(label, min_value, max_value, key=key, step=step,
                           on_change=_cb, args=(key, field), **kw)


def select(label, field, key, options, default=None, **kw):
    d = default if default is not None else options[0]
    _seed(key, field, d)
    if st.session_state.get(key) not in options:
        st.session_state[key] = d
    return st.selectbox(label, options, key=key, on_change=_cb, args=(key, field), **kw)


def check(label, field, key, default=False, **kw):
    _seed(key, field, default)
    return st.checkbox(label, key=key, on_change=_cb, args=(key, field), **kw)


def text(label, field, key, default="", **kw):
    _seed(key, field, default)
    return st.text_input(label, key=key, on_change=_cb, args=(key, field), **kw)
