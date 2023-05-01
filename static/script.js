$(document).ready(function() {

  $('#sample1').on('click', function(event){
    $('#query').val(document.getElementById("sample1").innerText)
  });
  $('#sample2').on('click', function(event){
    $('#query').val(document.getElementById("sample2").innerText)
  });
  $('#sample3').on('click', function(event){
    $('#query').val(document.getElementById("sample3").innerText)
  });
  $('#sample4').on('click', function(event){
    $('#query').val(document.getElementById("sample4").innerText)
  });

  $('#search-form').on('submit', function(event) {
    event.preventDefault();
    $('#results').empty();
    $('#loading').show();
    $.ajax({
      data: {
        query: $('#query').val(),
        num_results: $('#num-results').val()
      },
      type: 'GET',
      url: '/search'
    }).done(function(data) {
      $('#loading').hide();
      if (data.length === 0) {
        $('#results').append('<p>No results found.</p>');
      } else {
          $('#results').append('<div class="card"><div class="card-body"><p class="card-text">' + data + '</p></div></div>');
      }
    }).fail(function() {
      $('#loading').hide();
      $('#results').append('<p>An error occurred while processing your request.</p>');
    });
  });
});



