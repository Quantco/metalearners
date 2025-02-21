document.addEventListener("DOMContentLoaded", function () {
    Array.from(document.links)
        .filter(link => link.hostname != window.location.hostname)
        .forEach(link => link.target = '_blank');
    console.log(`${document.links.length} External links will be opened in a new tab after clicking.`);
});
