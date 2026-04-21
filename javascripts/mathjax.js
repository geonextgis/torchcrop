document$.subscribe(() => {
    if (window.MathJax) {
        MathJax.typesetPromise();
    }
});
