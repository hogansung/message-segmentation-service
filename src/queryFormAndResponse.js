import axios from "axios";
import React from 'react';

class QueryFormAndResponse extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            input_string: '',
            debug_mode: false,
            response_entities: [],
            overall_score: null,
            debug_logs: []
        };

        this.handleInputStringChange = this.handleInputStringChange.bind(this);
        this.handleDebugModeCheckboxChange = this.handleDebugModeCheckboxChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
    }

    handleInputStringChange(event) {
        this.setState({input_string: event.target.value});
    }

    handleDebugModeCheckboxChange(event) {
        this.setState({debug_mode: event.target.checked});
    }

    handleSubmit(event) {
        axios.post("/query", {
            "message": this.state.input_string,
            "debug_mode": this.state.debug_mode
        })
            .then(response => {
                // console.log(response.data)
                // console.log(response.data.response_entities)
                this.setState({
                    response_entities: response.data.response_entities,
                    overall_score: response.data.overall_score,
                    debug_logs: response.data.debug_logs,
                });
            }).catch((error) => {
                if (error.response) {
                    console.log(error.response)
                    console.log(error.response.status)
                    console.log(error.response.headers)
                }
            })

        event.preventDefault();
    }

    render() {
        return (
            <div className="bd-content ps-lg-2">
                <div>
                    <form onSubmit={this.handleSubmit}>
                        <div className="mb-3">
                            <label htmlFor="inputString" className="form-label"><b>Input String</b></label>
                            <input type="text" className="form-control" id="inputString" autoComplete="off"
                                spellCheck="false" onChange={this.handleInputStringChange}/>
                        </div>
                        <div className="form-check">
                           <input type="checkbox" className="form-check-input" id="debugModeCheckbox"
                               onClick={this.handleDebugModeCheckboxChange.bind(this)}/>
                           <label className="form-check-label" htmlFor="debugModeCheckbox">Debug Mode</label>
                       </div>
                       <br/>
                        <button type="submit" className="btn btn-primary">Submit</button>
                    </form>
                </div>
                <br/>
                <br/>
                <br/>
                <table className="table table-hover">
                    <thead>
                        <tr>
                            <th scope="col" style={{ width: "10%" }}>#</th>
                            <th scope="col" style={{ width: "20%" }}>Score</th>
                            <th scope="col" style={{ width: "70%" }}>Segment</th>
                        </tr>
                    </thead>
                    <tbody>
                        {
                            this.state.response_entities.map((response_entity, index) => (
                                <tr key={ index }>
                                    <th scope="row">{ index }</th>
                                    <td>{ response_entity.score.toFixed(5) }</td>
                                    <td>{ response_entity.segment }</td>
                                </tr>
                            ))
                        }
                    </tbody>
                </table>
                {
                    this.state.overall_score &&
                    <div>
                        <br/>
                        <br/>
                        <div className="card">
                            <div className="card-header">
                                <b>Overall Segments [ Score: { this.state.overall_score.toFixed(5) }]</b>
                            </div>
                            <div className="card-body">
                                <p className="card-title">{
                                    this.state.response_entities.map(response_entity => (
                                        response_entity.segment
                                    )).join(' ')
                                }</p>
                            </div>
                        </div>
                    </div>
                }
                {
                    this.state.debug_mode && this.state.debug_logs &&
                    <div>
                        <br/>
                        <br/>
                        <div className="card">
                            <div className="card-header">
                                <b>Debug Logs</b>
                            </div>
                            <div className="card-body">
                                {
                                    this.state.debug_logs.map((debug_log, _) => (
                                        <p>{ debug_log }</p>
                                    ))
                                }
                            </div>
                        </div>
                    </div>
                }
            </div>
        );
    }
}

export default QueryFormAndResponse;
