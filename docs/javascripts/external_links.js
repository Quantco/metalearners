// Modified from https://stackoverflow.com/a/4425214
document.addEventListener("DOMContentLoaded", function () {
    Array.from(document.links)
        .filter(link => link.hostname != window.location.hostname)
        .forEach(link => link.target = '_blank');
});
