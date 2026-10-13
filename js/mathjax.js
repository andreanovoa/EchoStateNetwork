// MathJax 3: arithmatex pages use \( \) and \[ \]; the rendered notebooks keep $ $ and $$ $$.
// pre and code are skipped by default, so $...$ inside code cells stays literal.
window.MathJax = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
};
