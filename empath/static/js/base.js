$(function () {
  $('html').addClass('render');

  // 부트스트랩 툴팁 정의
  const tooltipTriggerList = document.querySelectorAll('[data-bs-hover="tooltip"]');
  const tooltipList = [...tooltipTriggerList].map((tooltipTriggerEl) => new bootstrap.Tooltip(tooltipTriggerEl));

  // navbar
  const topPath = getTopPath();
  const navLinks = $('#navbar .nav-link');
  navLinks.each(function () {
    if ($(this).attr('href') === topPath) {
      $(this).addClass('active');
    } else {
      $(this).removeClass('active');
    }
  });
});

function getTopPath() {
  const currentUrl = window.location.href;
  const rootUrl = window.location.origin;
  const currentPath = currentUrl.substring(rootUrl.length);
  const topPath = currentPath.substring(0, currentPath.indexOf('/', 1) + 1);
  return topPath;
}
