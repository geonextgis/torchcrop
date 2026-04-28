document$.subscribe(() => {
    const renderMath = () => {
        if (window.MathJax && MathJax.typesetPromise) {
            MathJax.typesetPromise();
        } else {
            setTimeout(renderMath, 100);
        }
    };
    renderMath();
});
