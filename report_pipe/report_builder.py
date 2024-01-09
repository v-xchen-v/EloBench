class ReportBuilder:
    def __init__(self, result_dir: str, save_dir: str, report_pipes = []):
        self.result_dir = result_dir
        self.report_pipes = report_pipes
        self.save_dir = save_dir
    
    def load_data(self):
        for report_pipe in self.report_pipes:
            report_pipe.load_data(self.result_dir)

    def build(self):
        for report_pipe in self.report_pipes:
            report_pipe.build()
            # if hasattr(report_pipe, 'plot'):
            #     report_pipe.plot(self.save_dir)
            
    def plot(self):
        for report_pipe in self.report_pipes:
            if hasattr(report_pipe, 'plot'):
                report_pipe.plot(self.save_dir)
            