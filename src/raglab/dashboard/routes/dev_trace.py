"""The developer's step-by-step checkout of one widget thread.

Exists only while `RAGLAB_DEV_KEY` is set — a 404 otherwise, because a page for
developers should not announce itself to readers. The key is typed on the page
and never read from the address bar; what a browser keeps afterwards is a
random token this process remembers. The page itself is built by
`dev_trace_page`; these three routes are the lock on its door.
"""
from urllib.parse import parse_qs

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from raglab.dashboard import dev_trace_page


def register(app, context) -> None:

    def _trace_page(page: str) -> HTMLResponse:
        # A checkout window shows the thread as it is now, never as the
        # browser last saw it.
        return HTMLResponse(page, headers={'Cache-Control': 'no-store'})

    @app.get('/dev/trace', response_class=HTMLResponse)
    def dev_trace(request: Request, thread: str = ''):
        """The developer's step-by-step checkout of one widget thread — see
        `dev_trace_page`. Exists only while `RAGLAB_DEV_KEY` is set (a 404
        otherwise, because a page for developers should not announce itself
        to readers); asks for the key on the page and never reads it from the
        address bar."""
        if not dev_trace_page.configured():
            raise HTTPException(404)
        if not dev_trace_page.unlocked(request.cookies.get(dev_trace_page.COOKIE)):
            return _trace_page(dev_trace_page.unlock_page(next_thread=thread.strip()))
        return _trace_page(dev_trace_page.thread(thread) if thread.strip()
                           else dev_trace_page.index())

    @app.post('/dev/trace', response_class=HTMLResponse)
    async def dev_trace_unlock(request: Request):
        """The plate's form. The body is read by hand (`parse_qs`) rather than
        through `Form`, which would pull in a multipart parser for one field.
        The key is compared and forgotten; what the browser keeps is a random
        token this process remembers, with the key written into nothing."""
        if not dev_trace_page.configured():
            raise HTTPException(404)
        form = parse_qs((await request.body()).decode('utf-8', 'replace'))
        key = (form.get('key') or [''])[0]
        next_thread = (form.get('next') or [''])[0].strip()
        if not dev_trace_page.allowed(key):
            return _trace_page(dev_trace_page.unlock_page(
                next_thread=next_thread, error='That key did not match.'))
        response = RedirectResponse(
            dev_trace_page.thread_href(next_thread) if next_thread else dev_trace_page.PATH,
            status_code=303, headers={'Cache-Control': 'no-store'})
        response.set_cookie(dev_trace_page.COOKIE, dev_trace_page.issue_token(),
                            httponly=True, samesite='strict', path=dev_trace_page.PATH)
        return response

    @app.post('/dev/trace/lock')
    def dev_trace_lock(request: Request):
        """Lock: forgets the token server-side and clears the cookie, so a
        browser that kept it is back at the plate."""
        dev_trace_page.revoke(request.cookies.get(dev_trace_page.COOKIE))
        response = RedirectResponse(dev_trace_page.PATH, status_code=303,
                                    headers={'Cache-Control': 'no-store'})
        response.delete_cookie(dev_trace_page.COOKIE, path=dev_trace_page.PATH)
        return response
