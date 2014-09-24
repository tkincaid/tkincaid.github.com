var IDE = {
  view: '',
  form: '',
  frame: '',
  frame_loading: 'html',
  isDisplayed: function() {
    return casper.visible(this.view);
  },
  isFormHidden: function() {
    return casper.exists(this.form) && !casper.visible(this.form);
  },
  frameLoading: function(test) {
    // test.assertExists(this.frame_loading);
    // TODO: Wait for it to be loaded
    test.skip(2, "Issues working inside of frame");
  }
};
module.exports = function(){
  return IDE;
};