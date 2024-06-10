// static/script.js
$(document).ready(function() {
    $('#uploadForm').on('submit', function(event) {
        event.preventDefault();
        var formData = new FormData(this);
        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                $('#pdfText').text(response.text);
            }
        });
    });

    $('#questionForm').on('submit', function(event) {
        event.preventDefault();
        var question = $('#question').val();
        var context = $('#pdfText').text();
        $.ajax({
            url: '/ask',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ question: question, context: context }),
            success: function(response) {
                $('#answer').text(response.answer);
            }
        });
    });
});
