document$.subscribe(() => {
    if (!window.MathJax) return;

    MathJax.typesetClear();

    setTimeout(() => {
        MathJax.typesetPromise();
    }, 100);
});
