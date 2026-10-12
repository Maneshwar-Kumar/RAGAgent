"""MotoMedic rider UI on the RAGAgent backend: streamlit run rider_ui/app.py

The layout, copy and theme follow the MotoMedic rider UI
(github.com/Maneshwar-Kumar/motomedic-ai, rider_ui/). Behind it, the Google ADK agent in
bike_manual_agent/ answers from the selected manual via rider_ui/backend.py.
"""

import html
import logging
import queue
import re
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rider_ui import backend  # noqa: E402
from rider_ui.guards import LIMITS, Guard, check_question  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("rider_ui")

st.set_page_config(page_title='MotoMedic · Rider assistant', page_icon='🏍️', layout='wide')
st.markdown(Path(__file__).with_name('theme.css').read_text(encoding='utf-8'), unsafe_allow_html=True)

LABELS = {'answered': 'Answer', 'need_info': 'One detail to confirm',
          'not_covered': 'No supported answer', 'guardrail': 'No supported answer',
          'error': 'Unable to verify right now'}
SAMPLE_PHOTO = Path(__file__).resolve().parents[1] / 'samples' / 'white_smoke.jpg'
SAMPLE_QUESTION = 'Why is this smoke coming from my bike?'
REQUEST_TIMEOUT_S = 240
TIMEOUT_TEXT = ('MotoMedic took too long to check the manual, so no answer was given. '
                'Please try again, or start a new conversation.')


@st.cache_resource
def guard():
    return Guard(LIMITS)


def service():
    return backend.get_backend()


def photo_mime(data):
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    return None


def md_escape(text):
    return re.sub(r'([\\`*_{}\[\]()#+\-.!|>~$])', r'\\\1', text)


def safe_markdown(text):
    """Model text as markdown: HTML stays escaped (Streamlit default); links, images and
    LaTeX are neutralised so an answer cannot load or link to anything."""
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'<(https?://[^>]+)>', r'\1', text)
    return text.replace('$', '\\$')


def visitor():
    # Cloud Run's front end appends the caller's address, so the last entry can't be spoofed.
    forwarded = st.context.headers.get('X-Forwarded-For', '') or ''
    address = forwarded.split(',')[-1].strip() or (getattr(st.context, 'ip_address', None) or '')
    return address or st.session_state.user_id


def reset():
    for k, v in [('messages', []), ('session_id', None), ('pending', None), ('notice', None),
                 ('photos_used', 0), ('conversation_manual', None)]:
        st.session_state[k] = v
    st.session_state.photo_key += 1


for k, factory in [('messages', list), ('session_id', lambda: None), ('pending', lambda: None),
                   ('notice', lambda: None), ('uploaded', list), ('photo_key', int),
                   ('upload_key', int), ('photos_used', int), ('conversation_manual', lambda: None),
                   ('user_id', lambda: uuid.uuid4().hex)]:
    if k not in st.session_state:
        st.session_state[k] = factory()

try:
    manuals = service().manuals() + list(st.session_state.uploaded)
except Exception:
    log.exception('manual catalogue unavailable')
    manuals = []
if not manuals:
    st.error('No manuals are indexed yet. Run `python scripts/fetch_manuals.py --ingest`, then reload.')
    st.stop()
by_id = {m.manual_id: m for m in manuals}
if st.session_state.get('select_after_upload') in by_id:
    st.session_state.manual_select = st.session_state.pop('select_after_upload')
if st.session_state.get('manual_select') not in by_id:
    st.session_state.manual_select = manuals[0].manual_id
started = bool(st.session_state.messages or st.session_state.pending)

with st.sidebar:
    st.markdown('<div class="brand-mark">M<span>+</span></div><h2>MotoMedic</h2><p class="sidebar-subtitle">Your questions answered, backed by the manual.</p>', unsafe_allow_html=True)
    st.divider()
    st.caption('YOUR MOTORCYCLE')
    manual_id = st.selectbox('Motorcycle', list(by_id), key='manual_select', disabled=started,
                             format_func=lambda mid: by_id[mid].select_label + (' · uploaded' if by_id[mid].uploaded else ''))
    manual = by_id[st.session_state.conversation_manual or manual_id]
    confirmed = st.checkbox(manual.confirm_label, key='confirm_' + manual.manual_id, disabled=started)
    with st.expander('Add your bike’s manual (PDF)'):
        st.caption(f'Any owner’s or service manual with selectable text · up to {LIMITS.manual_megabytes} MB '
                   f'and {LIMITS.manual_pages} pages. Indexing takes about a minute.')
        pdf = st.file_uploader('Manual PDF', type=['pdf'], accept_multiple_files=False,
                               max_upload_size=LIMITS.manual_megabytes, label_visibility='collapsed',
                               key='manual_upload_' + str(st.session_state.upload_key), disabled=started)
        if pdf is not None and st.button('Index this manual', type='primary', use_container_width=True,
                                         disabled=started):
            refusal = guard().admit_upload()
            if refusal:
                st.error(refusal)
            else:
                try:
                    with st.spinner('Indexing your manual… about a minute for 150 pages'):
                        added = service().add_manual(pdf.getvalue(), pdf.name, LIMITS)
                    st.session_state.uploaded = [u for u in st.session_state.uploaded
                                                 if u.manual_id != added.manual_id] + [added]
                    st.session_state.select_after_upload = added.manual_id
                    st.session_state.upload_key += 1
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
                except Exception:
                    log.exception('manual upload failed')
                    st.error('The manual could not be indexed right now. Please try again.')
    st.caption('SELECTED SOURCE')
    st.markdown('**' + md_escape(manual.source_title) + '**')
    st.caption(md_escape(manual.edition_label))
    with st.expander('Manual details'):
        st.write(f'{manual.pages} pages, indexed into {manual.chunks} searchable passages. '
                 'Citations identify the printed page and PDF page of each passage used.')
        if manual.uploaded:
            st.caption('Uploaded in this browser session and visible only to your conversations.')
        elif manual.source_url:
            st.caption('Official download: ' + manual.source_url)
    st.divider()
    st.button('Start a new conversation', on_click=reset, use_container_width=True,
              disabled=bool(st.session_state.pending), help='Clear this chat and start a new conversation.')
    if manual.official_site and manual.brand:
        st.caption('Unable to find a resolution? Official service support remains available.')
        st.link_button('Contact ' + manual.brand + ' ↗', manual.official_site, use_container_width=True)

left, right = st.columns([5, 2])
with left:
    st.markdown('<div class="eyebrow">MOTOMEDIC / RIDER SUPPORT</div>', unsafe_allow_html=True)
    st.title('MotoMedic handles your routine queries for your ' + manual.short_name)
    st.write('Ask a question about your motorcycle. MotoMedic answers when the selected manual supports it, and tells you when it cannot.')
with right:
    st.markdown('<div class="source-pill">●&nbsp; ' + html.escape(manual.pill_label) + '<br><small>'
                + html.escape(manual.edition_short) + '</small></div>', unsafe_allow_html=True)
st.divider()

ready = confirmed and not st.session_state.pending


def photo_uploader():
    st.file_uploader('Add a bike photo', type=['jpg', 'jpeg', 'png', 'webp'], accept_multiple_files=False,
                     max_upload_size=LIMITS.photo_megabytes, key='photo_' + str(st.session_state.photo_key),
                     disabled=not ready)
    st.caption(f'Photos: one per question, up to {LIMITS.photo_megabytes} MB each; '
               f'{LIMITS.photos_per_conversation} per conversation. MotoMedic describes what it sees, '
               'then checks the manual.')


def submit(text, sample=False):
    """Validate and send one question; the answer is collected by the pending block below."""
    text = text.strip()
    st.session_state.notice = check_question(text)
    if st.session_state.notice:
        return
    photo = None
    if sample:
        photo = (SAMPLE_PHOTO.read_bytes(), 'image/jpeg')
    else:
        upload = st.session_state.get('photo_' + str(st.session_state.photo_key))
        if upload is not None:
            data = upload.getvalue()
            mime = photo_mime(data)
            if mime is None:
                st.session_state.notice = 'Photos must be JPEG, PNG or WebP images.'
                return
            photo = (data, mime)
    if photo and st.session_state.photos_used >= LIMITS.photos_per_conversation:
        st.session_state.notice = (f'This conversation already has {LIMITS.photos_per_conversation} photos. '
                                   'Start a new conversation to add another.')
        return
    turns = sum(1 for role, _ in st.session_state.messages if role == 'rider')
    if turns >= LIMITS.turns_per_conversation:
        st.session_state.notice = 'This chat is long. Start a new conversation for your next question.'
        return
    st.session_state.notice = guard().admit_question(visitor())
    if st.session_state.notice:
        return
    try:
        if not st.session_state.session_id:
            st.session_state.session_id = service().start_conversation(st.session_state.user_id, manual)
            st.session_state.conversation_manual = manual.manual_id
        progress = queue.Queue()
        future = service().ask(st.session_state.user_id, st.session_state.session_id, text, photo, progress)
    except Exception:
        log.exception('could not start the question')
        st.session_state.notice = 'MotoMedic is unavailable right now. Please try again in a moment.'
        return
    st.session_state.messages.append(('rider', {'text': text, 'photo': photo[0] if photo else None}))
    st.session_state.pending = {'future': future, 'progress': progress, 'started': time.monotonic()}
    if photo:
        st.session_state.photos_used += 1
        st.session_state.photo_key += 1
    st.rerun()


if not st.session_state.messages and not st.session_state.pending:
    with st.container(border=True):
        st.markdown('### What would you like to know?')
        st.write('Ask your question in up to 100 words. Include relevant details and anything you have already checked.')
        st.caption('Up to 2,000 characters per message.')
        query = st.chat_input('Ask a question…', key='symptom_input', max_chars=LIMITS.message_characters,
                              disabled=not ready)
        photo_uploader()
        if st.button('Try the sample photo · white smoke from the exhaust', disabled=not ready):
            submit(SAMPLE_QUESTION, sample=True)
        if not confirmed:
            st.caption('Confirm your motorcycle in the sidebar to start.')
        if query and query.strip():
            submit(query)
        if st.session_state.notice:
            st.warning(st.session_state.notice)
    cols = st.columns(3)
    for col, title, body in zip(cols, ['01 · Ask', '02 · Read', '03 · Continue'],
                               ['Ask about your motorcycle, with a photo if it helps.',
                                'Read a supported answer with source references.',
                                'Ask another question when you are ready.']):
        with col:
            with st.container(border=True):
                st.markdown('**' + title + '**')
                st.write(body)

for role, value in st.session_state.messages:
    with st.chat_message('user' if role == 'rider' else 'assistant', avatar='🏍️' if role == 'rider' else '🛠️'):
        if role == 'rider':
            if value.get('photo'):
                st.image(value['photo'], width=260)
            st.text(value['text'])
            continue
        answer = value
        st.caption(LABELS.get(answer.status, 'Answer'))
        if answer.status == 'error':
            st.warning('Unable to verify right now')
        st.markdown(safe_markdown(answer.body))
        if answer.citations:
            with st.expander('Source · ' + ', '.join(c.label for c in answer.citations)):
                st.caption(md_escape(manual.name + ' · ' + manual.edition_label))
                for c in answer.citations:
                    st.text(c.label + ' · PDF page ' + str(c.pdf_page) + (' · ' + c.section if c.section else ''))
                    st.markdown('<div class="source-excerpt">' + html.escape(c.excerpt) + '</div>',
                                unsafe_allow_html=True)
        details = []
        if answer.searches:
            details.append('Checked the manual for: ' + '; '.join('“' + md_escape(q) + '”' for q in answer.searches))
        if answer.elapsed_s:
            details.append(f'{answer.elapsed_s:.1f} s')
        if details:
            st.caption(' · '.join(details))
        if answer.reference:
            st.caption('Response reference: ' + md_escape(answer.reference))

pending = st.session_state.pending
if pending:
    with st.chat_message('assistant', avatar='🛠️'):
        slot = st.empty()
        latest = 'Connecting to MotoMedic…'
        with st.spinner('Checking the manual…'):
            while not pending['future'].done():
                while not pending['progress'].empty():
                    latest = pending['progress'].get_nowait()
                slot.caption(latest)
                if time.monotonic() - pending['started'] > REQUEST_TIMEOUT_S:
                    break
                time.sleep(0.25)
    if pending['future'].done():
        try:
            result = pending['future'].result()
        except Exception:
            log.exception('answer failed')
            result = backend.Answer('error', backend.ERROR_TEXT, (), (), 0.0, '')
    else:
        pending['future'].cancel()
        result = backend.Answer('error', TIMEOUT_TEXT, (), (), 0.0, '')
    st.session_state.messages.append(('assistant', result))
    st.session_state.pending = None
    st.rerun()

if st.session_state.messages:
    with st.expander('What this conversation remembers'):
        questions = [v['text'] for r, v in st.session_state.messages if r == 'rider']
        for q in questions:
            st.text('• ' + q)
        searches = [q for r, v in st.session_state.messages if r == 'assistant' for q in v.searches]
        if searches:
            st.caption('Manual searches so far: ' + '; '.join('“' + md_escape(q) + '”' for q in searches))
        st.caption('Answers come only from the selected manual. Reported success does not verify the repair or identify its cause.')
    if st.session_state.notice:
        st.warning(st.session_state.notice)
    with st.expander('Add a bike photo to your next question'):
        photo_uploader()
    st.caption('Up to 100 words per question.')
    query = st.chat_input('Ask another question…', key='followup_input', max_chars=LIMITS.message_characters,
                          disabled=not ready)
    if query and query.strip():
        submit(query)
        st.rerun()
st.markdown('<div class="footer-note">Built on the Owner\'s Manual, with no misleading instructions on the internet</div>', unsafe_allow_html=True)
